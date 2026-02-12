import os
import jax
import jax.numpy as jnp
from iree import runtime as ireert
from iree.compiler import compile_str
import collections
from collections import OrderedDict
from collections.abc import Mapping
import time

_STATS = {"total_ms": 0.0, "transfer_ms": 0.0, "post_ms": 0.0, "count": 0}
_DEVICE_PARAMS_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_DEVICE_PARAMS_CACHE_MAX = 2

# --- 1. The Helper provided in your prompt ---
def load_mlir_from_file(mlir_path, backend_name="cpu"):
    """Compiles or loads a cached IREE module from an MLIR file."""
    if backend_name == "cpu":
        target_backends = ["llvm-cpu"]
    elif backend_name == "gpu":
        target_backends = ["cuda"]
    else:
        raise ValueError("Unsupported backend name. Use 'cpu' or 'gpu'.")

    binary_path = mlir_path + ".bin"

    if os.path.exists(binary_path):
        print(f"✅ Loading compiled flatbuffer from cache: {binary_path}")
        with open(binary_path, "rb") as f:
            flatbuffer_blob = f.read()
    else:
        print(f"⏳ Compiling MLIR file, no cache found for: {mlir_path}")
        with open(mlir_path, "r") as f:
            mlir_module = f.read()

        # Compile to IREE bytecode
        flatbuffer_blob = compile_str(
            mlir_module,
            target_backends=target_backends,
            input_type="stablehlo",
            extra_args=["--iree-cuda-target=sm_80", "--iree-cuda-target-features=+ptx76"] 
            if backend_name == "gpu" else []
        )

        with open(binary_path, "wb") as f:
            print(f"Writing compiled flatbuffer to cache: {binary_path}")
            f.write(flatbuffer_blob)

    config = ireert.Config(target_backends[0])
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
    ctx.add_vm_module(vm_module)
    # print(ctx.modules)
    return ctx.modules, ctx.config.device

# --- 2. Helper Class to mimic the output of model.apply ---
# gemma code expects an object with .logits and .cache attributes
InferenceOutput = collections.namedtuple("InferenceOutput", ["logits", "cache"])

def _normalize_params(params):
    if params is None:
        return None
    if isinstance(params, Mapping) and "params" in params:
        return params["params"]
    if hasattr(params, "params") and not hasattr(params, "dtype"):
        try:
            return params.params
        except Exception:
            return params
    if isinstance(params, Mapping) and "vision_encoder" in params:
        params = dict(params)
        params.pop("vision_encoder", None)
    if isinstance(params, Mapping) and "embedder" in params:
        embedder = params.get("embedder")
        if isinstance(embedder, Mapping) and "input_embedding" in embedder:
            embedder = dict(embedder)
            embedder = {"input_embedding": embedder["input_embedding"]}
            params = dict(params)
            params["embedder"] = embedder
    return params


def _params_signature(params):
    params = _normalize_params(params)
    if params is None:
        return None
    flat_params, treedef = jax.tree_util.tree_flatten(params)
    # Use treedef string + shapes + dtypes for a stable cache key.
    shapes = tuple((tuple(p.shape), str(p.dtype)) for p in flat_params)
    # PyTreeDef doesn't expose to_string() on some JAX versions.
    return (str(treedef), shapes)

class GemmaIreeBridge:
    def __init__(self, prefill_mod, decode_mod, device):
        """
        Loads the prefill and decode artifacts and prepares them for execution.
        """
        self.prefill_module = prefill_mod
        self.decode_module = decode_mod
        self.device = device
        self._host_params_ref = None
        self._host_params_norm = None
        self._device_params = None

    def _to_jax(self, device_array):
        """Zero-copy (if possible) or efficient transfer from IREE to JAX Array."""
        # For now, we cast via host. In optimized setups, use dlpack.
        return jnp.array(device_array)

    def _params_to_device(self, params):
        params_input = params
        params = _normalize_params(params)
        if params is None:
            raise ValueError("params must be provided for IREE execution.")
        sig = _params_signature(params_input)
        if sig in _DEVICE_PARAMS_CACHE:
            _DEVICE_PARAMS_CACHE.move_to_end(sig)
            self._host_params_ref = params_input
            self._host_params_norm = params
            self._device_params = _DEVICE_PARAMS_CACHE[sig]
            return self._device_params
        flat_params, _ = jax.tree_util.tree_flatten(params)
        device_params = [
            ireert.asdevicearray(self.device, p) for p in flat_params
        ]
        _DEVICE_PARAMS_CACHE[sig] = device_params
        _DEVICE_PARAMS_CACHE.move_to_end(sig)
        if len(_DEVICE_PARAMS_CACHE) > _DEVICE_PARAMS_CACHE_MAX:
            _DEVICE_PARAMS_CACHE.popitem(last=False)
        self._host_params_ref = params_input
        self._host_params_norm = params
        self._device_params = device_params
        return device_params

    def _run_module(self, module_function, params, tokens, positions, mask, cache):
        """
        Generic handler for both prefill and decode.
        1. Flattens inputs (Params, Tokens, Pos, Mask, CachePyTree)
        2. Calls IREE (gets f32)
        3. Unflattens outputs (Logits, CachePyTree) and casts back to bf16
        """
        
        global _STATS
        
        # --- A. Flatten Inputs ---
        args_structure = (tokens, positions, mask, cache)
        
        # --- B. EXPLICIT DATA MOVEMENT (Host -> Device) ---
        # Timer starts: Capturing Host-to-Device transfer
        t_transfer_start = time.perf_counter()
        flat_args, _ = jax.tree_util.tree_flatten(args_structure)
        device_params = self._params_to_device(params)
        
        device_args = device_params + [
            ireert.asdevicearray(self.device, arg) for arg in flat_args
        ]
        
        t_transfer_end = time.perf_counter()

        # --- C. Call IREE (Compute Only) ---
        t_compute_start = time.perf_counter()
        
        # Pass the allocated device arrays directly
        results_flat = module_function(*device_args)
        
        t_compute_end = time.perf_counter()
        
        # If the function returns a single item, IREE might not wrap it in a tuple/list.
        # Ensure it is a list for unflattening.
        if not isinstance(results_flat, (list, tuple)):
            results_flat = [results_flat]

        # --- C. Unflatten Outputs & Cast back to BF16 ---
        
        # results_flat[0] -> logits (f32)
        # results_flat[1:] -> flattened cache leaves (f32 or i32)
        
        logits_buffer_f32 = results_flat[0]
        cache_buffers_f32 = results_flat[1:]
        
        # Convert IREE f32 buffers back to JAX f32 arrays
        logits_f32 = self._to_jax(logits_buffer_f32)
        cache_leaves_f32 = [self._to_jax(b) for b in cache_buffers_f32]

        # --- START F32 TO BF16 CONVERSION ---
        # Helper to cast f32 leaves back to bf16
        def _cast_leaf_to_bf16(leaf):
            if hasattr(leaf, 'dtype') and leaf.dtype == jnp.float32:
                return leaf.astype(jnp.bfloat16)
            return leaf

        # Cast logits back to bf16
        logits_bf16 = _cast_leaf_to_bf16(logits_f32)
        
        # Unflatten the cache (which is still f32/i32)
        _, cache_treedef = jax.tree_util.tree_flatten(cache)
        updated_cache_f32 = jax.tree_util.tree_unflatten(cache_treedef, cache_leaves_f32)
        
        # Cast the entire unflattened cache tree back to bf16
        updated_cache_bf16 = jax.tree_util.tree_map(_cast_leaf_to_bf16, updated_cache_f32)
        
        t_end = time.perf_counter()
        
        _STATS["transfer_ms"] += (t_transfer_end - t_transfer_start) * 1000
        _STATS["total_ms"] += (t_end - t_transfer_start) * 1000
        _STATS["post_ms"] += (t_end - t_compute_end) * 1000
        _STATS["count"] += 1
        
        
        return InferenceOutput(logits=logits_bf16, cache=updated_cache_bf16)
        # --- END F32 TO BF16 CONVERSION ---

    def run_prefill(self, tokens, positions, attention_mask, cache, params=None, **kwargs):
        """
        Replacement for model.apply in the prefill stage.
        Note: params is required to avoid baking weights into the MLIR.
        """
        # print('!' * 20)
        # Invoke the main function of the prefill module
        return self._run_module(
            self.prefill_module.main, 
            params, tokens, positions, attention_mask, cache
        )

    def run_decode(self, tokens, positions, attention_mask, cache, params=None, **kwargs):
        """
        Replacement for model.apply in the decode/sampler stage.
        """
        return self._run_module(
            self.decode_module.main, 
            params, tokens, positions, attention_mask, cache
        )
        
        
_iree_runner_cache = {}

def get_iree_runner(
    params,
    prefill_path="gemma_4b_2048_prefill.mlir", 
    decode_path="gemma_4b_2048_decode.mlir",
    backend="gpu"
):
    if params is None:
        raise ValueError("params must be provided to create a per-model IREE runner.")
    cache_key = (prefill_path, decode_path, backend)
    runner = _iree_runner_cache.get(cache_key)
    if runner is None:
        print(f"Initializing IREE Bridge with backend: {backend}")
        # Helper now returns (modules, device)
        prefill_ctx_mods, device = load_mlir_from_file(prefill_path, backend)
        decode_ctx_mods, _ = load_mlir_from_file(decode_path, backend)
        prefill_mod = prefill_ctx_mods.jit_prefill_forward
        decode_mod = decode_ctx_mods.jit_decode_forward
        runner = GemmaIreeBridge(prefill_mod, decode_mod, device)
        _iree_runner_cache[cache_key] = runner
    return runner


def show_stats():
    global _STATS
    if _STATS["count"] == 0:
        print("[Profile] No runs recorded.")
        return

    avg_total = _STATS["total_ms"] / _STATS["count"]
    avg_transfer = _STATS["transfer_ms"] / _STATS["count"]
    avg_post = _STATS["post_ms"] / _STATS["count"]
    avg_compute = avg_total - avg_transfer - avg_post
    
    print(f"[Profile] Run {_STATS['count']}")
    print(f"  Avg Total:    {avg_total:.2f} ms")
    print(f"  Avg Transfer: {avg_transfer:.2f} ms (Host->Device)")
    print(f"  Avg Compute:  {avg_compute:.2f} ms")
    print(f"  Avg Post:     {avg_post:.2f} ms")
