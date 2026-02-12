import os
import argparse
import logging
import jax
import jax.numpy as jnp

from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

from jax import export
import os, certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# --- Gemma Imports ---
from gemma import gm
from gemma import peft
# No longer using the generic tokenizer import

# Disable verbose logging and set up JAX
logging.disable(logging.WARNING)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ['JAX_PLATFORMS'] = 'cpu'

def _dtype(params) -> jnp.dtype:
  return jax.tree.leaves(params)[0].dtype

def _to_shape_dtype(tree):
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree
    )

class ApproxMLIRGenerator:
    """A concise class to compile and run Gemma with IREE."""

    def __init__(self, size, max_length=None):
        """Initialize the Gemma runner."""
        assert max_length
        self.max_length = max_length
        self.size = size
        
        if size == "270m":
            print("Loading Gemma 270M model")
            self.model = gm.nn.Gemma3_270M(dtype=jnp.float16)
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
        elif size == "1b":
            print("Loading Gemma 1B model")
            self.model = gm.nn.Gemma3_1B(dtype=jnp.float16)
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
        elif size == "4b":
            print("Loading Gemma 4B model ")
            self.model = gm.nn.Gemma3_4B(dtype=jnp.float16)
            self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
        elif size == "4b-q":
            print("Load INT8 Gemma 4B model")
            self.model = gm.nn.IntWrapper(model = gm.nn.Gemma3_4B(), dtype = jnp.int8)
            parms_noq = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
            self.params = peft.quantize(parms_noq, method=peft.QuantizationMethod.INT8, in_place_keys=False, checkpoint_kernel_key="w")
        else: raise ValueError(f"Unsupported model size: {size}")
        
        print(f"Set max sequence length to: {self.max_length}")

    def compile_prefill(self, batch_size=1, prefill_length=128, save_path=None):
        """Compiles the Prefill phase (Prompt Processing)."""
        if save_path is None:
            save_path = f"gemma_{self.size}_{self.max_length}_prefill"
        mlir_path = f"{save_path}.mlir"

        if os.path.exists(mlir_path):
            print(f"✅ Found existing Prefill MLIR: {mlir_path}")
            return mlir_path

        print(f"⏳ Compiling Prefill (Length {prefill_length}) to {mlir_path}...")

        # --- 1. Prepare Inputs for Prefill ---
        # Source [138]: Prefill tokens shape is [B, L]
        dummy_tokens = jax.ShapeDtypeStruct(
            (batch_size, prefill_length), jnp.int32
        )
        
        # Source [317]: Positions match token length
        dummy_positions = jax.ShapeDtypeStruct(
            (batch_size, prefill_length), jnp.int32
        )
        
        # Source [141]: Attention mask is [B, L, cache_length]
        # The mask allows the prefill tokens to attend to themselves
        dummy_mask = jax.ShapeDtypeStruct(
            (batch_size, prefill_length, self.max_length), jnp.bool_
        )

        # Source [280]: Init cache using the model's native method
        dummy_cache = self.model.init_cache(
            batch_size=batch_size,
            cache_length=self.max_length,
            dtype=_dtype(self.params)
        )
        dummy_cache = _to_shape_dtype(dummy_cache)
        dummy_params = _to_shape_dtype(self.params)

        # --- 2. Define Prefill Wrapper ---
        def prefill_forward(params, tokens, positions, mask, cache):
            # Source [194]: Prefill apply call
            # We use return_last_only=True to get just the last logit for the next token prediction
            out = self.model.apply(
                {'params': params},
                tokens=tokens,
                positions=positions,
                attention_mask=mask,
                cache=cache,
                return_last_only=True 
            )
            # --- START BF16 TO F32 FIX ---
            # Create a helper function to cast only bfloat16 leaves
            def _cast_leaf_to_f32(leaf):
                if hasattr(leaf, 'dtype') and leaf.dtype == jnp.bfloat16:
                    return leaf.astype(jnp.float32)
                return leaf

            # Cast logits
            logits_f32 = _cast_leaf_to_f32(out.logits)
            
            # Recursively iterate over the cache PyTree and cast all bf16 arrays
            cache_f32 = jax.tree_util.tree_map(_cast_leaf_to_f32, out.cache)
            
            return logits_f32, cache_f32
            # --- END BF16 TO F32 FIX ---

        # --- 3. Export ---
        print("Exporting Prefill...")
        jitted_fn = jax.jit(prefill_forward)
        exported = export.export(jitted_fn)(
            dummy_params, dummy_tokens, dummy_positions, dummy_mask, dummy_cache
        )

        with open(mlir_path, "w") as f:
            f.write(exported.mlir_module())
        
        print(f"✅ Saved Prefill MLIR to {mlir_path}\n")
        return mlir_path

    def compile_decode(self, batch_size=1, save_path=None):
        """Compiles the Decode phase (Token Generation)."""
        if save_path is None:
            save_path = f"gemma_{self.size}_{self.max_length}_decode"
        mlir_path = f"{save_path}.mlir"

        if os.path.exists(mlir_path):
            print(f"✅ Found existing Decode MLIR: {mlir_path}")
            return mlir_path

        print(f"⏳ Compiling Decode (1 Token) to {mlir_path}...")

        # --- 1. Prepare Inputs for Decode ---
        # Source [583]: Decode processes 1 token at a time [B, 1]
        dummy_tokens = jax.ShapeDtypeStruct((batch_size, 1), jnp.int32)
        dummy_positions = jax.ShapeDtypeStruct((batch_size, 1), jnp.int32)
        
        # Source [586]: Attention mask for decode is [B, 1, cache_length]
        dummy_mask = jax.ShapeDtypeStruct(
            (batch_size, 1, self.max_length), jnp.bool_
        )

        # Init cache (same structure as prefill)
        dummy_cache = self.model.init_cache(
            batch_size=batch_size,
            cache_length=self.max_length,
            dtype=_dtype(self.params)
        )
        dummy_cache = _to_shape_dtype(dummy_cache)
        dummy_params = _to_shape_dtype(self.params)

        # --- 2. Define Decode Wrapper ---
        def decode_forward(params, tokens, positions, mask, cache):
            # Source [581]: Decode apply call
            out = self.model.apply(
                {'params': params},
                tokens=tokens,
                positions=positions,
                attention_mask=mask,
                cache=cache,
            )
            # --- START BF16 TO F32 FIX ---
            # Create a helper function to cast only bfloat16 leaves
            def _cast_leaf_to_f32(leaf):
                if hasattr(leaf, 'dtype') and leaf.dtype == jnp.bfloat16:
                    return leaf.astype(jnp.float32)
                return leaf

            # Cast logits
            logits_f32 = _cast_leaf_to_f32(out.logits)
            
            # Recursively iterate over the cache PyTree and cast all bf16 arrays
            cache_f32 = jax.tree_util.tree_map(_cast_leaf_to_f32, out.cache)
            
            return logits_f32, cache_f32
            # --- END BF16 TO F32 FIX ---

        # --- 3. Export ---
        print("Exporting Decode...")
        jitted_fn = jax.jit(decode_forward)
        exported = export.export(jitted_fn)(
            dummy_params, dummy_tokens, dummy_positions, dummy_mask, dummy_cache
        )

        with open(mlir_path, "w") as f:
            f.write(exported.mlir_module())

        print(f"✅ Saved Decode MLIR to {mlir_path}\n")
        return mlir_path

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Compile and run Gemma with IREE')
    parser.add_argument('--compile-only', action='store_true', help='Only compile the model without running')
    parser.add_argument('--generate', type=str, help='Text prompt for multi-token generation (non-interactive)')
    parser.add_argument('--num-tokens', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature for generation')

    args = parser.parse_args()
    for max_length in [2048]:
        for model_size in ["1b", "4b"]:
            runner = ApproxMLIRGenerator(model_size, max_length)
            runner.compile_prefill(prefill_length=max_length)
            runner.compile_decode()

if __name__ == "__main__":
    main()
