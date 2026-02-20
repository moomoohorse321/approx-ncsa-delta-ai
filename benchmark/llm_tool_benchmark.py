#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Default env values (user-provided env vars override these).
_DEFAULT_CGEIST_PATH = "/work/nvme/beqg/PolygeistSample/build/bin/cgeist"
_DEFAULT_CGEIST_RESOURCE_DIR = "/work/nvme/beqg/PolygeistSample/llvm-project/build/lib/clang/18"
_DEFAULT_CGEIST_INCLUDE_DIR = "/work/nvme/beqg/PolygeistSample/tools/cgeist/Test/polybench/utilities"


def _set_default_env() -> None:
    os.environ.setdefault("CGEIST_PATH", _DEFAULT_CGEIST_PATH)
    os.environ.setdefault("CGEIST_RESOURCE_DIR", _DEFAULT_CGEIST_RESOURCE_DIR)
    os.environ.setdefault("CGEIST_INCLUDE_DIR", _DEFAULT_CGEIST_INCLUDE_DIR)


_set_default_env()

_LOG = logging.getLogger("llm_tool_bench")

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import approx_runtime as ar
import jax
import jax.numpy as jnp
from sentence_transformers import SentenceTransformer, util
import torch
from gemma.gm.iree import _iree as gemma_iree

# LLM + selector logic mirrors selector_tune_llm.py.
from llm import LLMManager

# Import LLM tool questions and grading.
LLM_TOOL_DIR = Path(__file__).resolve().parents[1] / "LLM_tool"
if str(LLM_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_TOOL_DIR))
from questions import Question, evalutation_questions, questions_to_run  # noqa: E402

# Import benchmark helpers from approx-runtime examples.
RUNTIME_BENCH_DIR = Path(
    os.environ.get(
        "APPROX_RUNTIME_BENCH_DIR",
        "/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime/examples/benchmark",
    )
)
if str(RUNTIME_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_BENCH_DIR))
from benchmark_common import (  # noqa: E402
    compile_cpp_path_to_annotated_mlir,
    compile_mlir_to_native_exec,
    default_toolchain,
    run_exec,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    cpp_path: Path
    command_parsing_fn: Callable[[List[str], "ToolRuntime"], List[str]]


class ToolRuntime:
    def __init__(
        self,
        name: str,
        annotated_mlir: str,
        params: Dict[str, ar.TunableParam],
        exec_cache: Dict[Tuple[Tuple[str, int], ...], Path],
        exec_root: Path,
    ) -> None:
        self.name = name
        self.annotated_mlir = annotated_mlir
        self.params = params
        self.exec_cache = exec_cache
        self.exec_root = exec_root


# Shared models for reranking
_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
_tokenizer = None


# Selector kernel (same as selector_tune_llm.py).

def get_state(necessity_score: jax.Array) -> jax.Array:
    return necessity_score


@ar.Knob(
    decision_tree=ar.DecisionTree(
        state_function=get_state,
        state_indices=[0],
        thresholds=[2],
        decisions=[0, 1],
        transform_type="task_skipping",
        thresholds_lower=[0],
        thresholds_upper=[5],
        decision_values=[0, 1],
    )
)
def selector_kernel(necessity_score: jax.Array) -> jax.Array:
    return jax.lax.cond(
        necessity_score < 2,
        lambda _: jnp.int32(0),
        lambda _: jnp.int32(1),
        operand=None,
    )


def build_selector_mlir() -> Tuple[str, List[str], Dict]:
    config = ar.get_config(selector_kernel)
    prompt_shape = jax.ShapeDtypeStruct((), jnp.int32)
    functions = {
        "selector_kernel": (selector_kernel, (prompt_shape,), config),
        "get_state": (get_state, (prompt_shape,), None),
    }
    mlir = ar.export_module_with_configs(functions)
    pipeline = ar.get_pipeline_for_config(config)
    return mlir, pipeline, config


def compile_selector(
    mlir_annotated: str,
    pipeline: List[str],
    backend: str,
    out_dir: Path,
    extra_args: List[str],
) -> Tuple[object, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled = ar.compile(mlir_annotated, passes=pipeline)
    vmfb = ar.compile_to_iree(
        compiled,
        backend=backend,
        input_type="stablehlo",
        extra_args=extra_args,
    )
    vmfb_path = out_dir / "selector.vmfb"
    vmfb_path.write_bytes(vmfb)
    modules, _ = ar.load_module(vmfb, backend=backend)
    return modules, vmfb_path


def _reset_gemma_stats() -> None:
    if hasattr(gemma_iree, "_STATS"):
        gemma_iree._STATS.update(
            {"total_ms": 0.0, "transfer_ms": 0.0, "post_ms": 0.0, "count": 0}
        )


def _get_total_compute_ms() -> float:
    stats = getattr(gemma_iree, "_STATS", None)
    if not stats or stats.get("count", 0) == 0:
        raise RuntimeError("No Gemma IREE stats recorded for total_compute")
    return stats["total_ms"] - stats["transfer_ms"] - stats["post_ms"]


def select_variant(modules, prompt_len: int) -> int:
    result = modules.module["selector_kernel"](jnp.array(prompt_len, dtype=jnp.int32))
    if hasattr(result, "to_host"):
        result = result.to_host()
    return int(result)


def _measure_llm_call(llm: LLMManager, prompt: str, model_size: str) -> Tuple[str, float]:
    _reset_gemma_stats()
    output = llm.generate(prompt, model_size)
    time_ms = _get_total_compute_ms()
    return output, time_ms


def truncate(text: str, context_len: int, padding: int = 128, reserved: int = 64) -> str:
    global _tokenizer
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialized")
    max_len = context_len - padding - reserved
    token_ids_list = _tokenizer.encode(text, add_bos=False)
    if len(token_ids_list) >= max_len:
        token_ids_list = token_ids_list[: -padding][: max_len - reserved] + token_ids_list[-padding:]
    return _tokenizer.decode(token_ids_list)


def rerank(docs_string: str, query: str) -> str:
    chunk_size = 512
    docs = docs_string.strip().split("\n")
    all_chunks = [doc[j : j + chunk_size] for doc in docs for j in range(0, len(doc), chunk_size) if doc]
    if not all_chunks:
        return ""
    query_embedding = _embedding_model.encode(query, convert_to_tensor=True, device=torch.device("cuda"))
    chunk_embeddings = _embedding_model.encode(all_chunks, convert_to_tensor=True, device=torch.device("cuda"))
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_chunks = [all_chunks[i] for i in sorted_indices]
    return "\n".join(sorted_chunks)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _kb_post_process(data_string: str, dataset_root: Path, question: str) -> str:
    pattern = re.compile(r"Rank\s+(\d+):\s+Doc\s+(\S+)\s+\(Score:\s+([\d\.]+)\)")
    parsed_data: List[Tuple[int, str, float]] = []
    for line in data_string.strip().split("\n"):
        match = pattern.match(line.strip())
        if match:
            rank = int(match.group(1))
            doc_id = match.group(2)
            score = float(match.group(3))
            parsed_data.append((rank, doc_id, score))

    content_file_path = dataset_root / "content.txt"
    doc_id_to_content: Dict[str, str] = {}
    for line in _read_text(content_file_path).splitlines():
        doc_id_str, stored_content = line.split("|", 1)
        doc_id_to_content[doc_id_str] = stored_content.strip()

    final_content_string = ""
    for rank, doc_id, _ in parsed_data:
        if rank > 5:
            continue
        stored_content = doc_id_to_content.get(doc_id, "Content not found for this doc_id")
        content = stored_content.replace("\\n", "\n")
        final_content_string += content + "\n"
    final_content_string = rerank(final_content_string, question)
    return final_content_string.strip()


def strip_quotes_from_args(args: List[str], _: "ToolRuntime") -> List[str]:
    return [arg.strip().strip("'\"") for arg in args]


def normalize_flag_equals_args(args: List[str], _: "ToolRuntime") -> List[str]:
    """Normalize CLI args like '-k=10' -> ['-k', '10']."""
    cleaned = strip_quotes_from_args(args, _)
    normalized: List[str] = []
    for token in cleaned:
        if token.startswith("-") and "=" in token:
            key, value = token.split("=", 1)
            normalized.append(key)
            if value:
                normalized.append(value)
        else:
            normalized.append(token)
    return normalized


def _hash_config(config: Dict[str, int], param_names: set[str]) -> Tuple[Tuple[str, int], ...]:
    items = tuple(sorted((k, v) for k, v in config.items() if k in param_names))
    return items


def _config_tag(config_key: Tuple[Tuple[str, int], ...]) -> str:
    digest = hashlib.sha1(repr(config_key).encode("utf-8")).hexdigest()
    return digest[:12]


def _build_tools(
    toolchain: ar.ToolchainConfig,
    cgeist_config: ar.CgeistConfig,
) -> Tuple[List[ToolSpec], Dict[str, ToolRuntime]]:
    tools = [
        ToolSpec(
            name="kmeans",
            description=(
                "Find K centroids of N nodes and related metadata (tool already knows input/output of questions). "
                "Example command=kmeans(-k,10,-n,100), meaning finding 10 centroids for 100 nodes."
            ),
            cpp_path=RUNTIME_BENCH_DIR / "src" / "approx_kmeans.c",
            command_parsing_fn=normalize_flag_equals_args,
        ),
        ToolSpec(
            name="lavaMD",
            description=(
                "Molecular dynamics simulation and related metadata (tool already knows input/output of questions). "
                "Example command=lavaMD(-boxes1d,10), meaning running a molecular dynamics simulation for a system with boxes1d = 10."
            ),
            cpp_path=RUNTIME_BENCH_DIR / "src" / "approx_lavaMD.c",
            command_parsing_fn=normalize_flag_equals_args,
        ),
        ToolSpec(
            name="bm25",
            description=(
                "search engine for wikipedia pages, tool for answering simple questions. "
                "Example command=bm25(), meaning searching wikipedia for relevant pages to answer a simple question."
            ),
            cpp_path=RUNTIME_BENCH_DIR / "src" / "approx_bm25.c",
            command_parsing_fn=strip_quotes_from_args,
        ),
        ToolSpec(
            name="kb",
            description=(
                "Knowledge base of wikipedia pages, tool for answering hard questions. "
                "Example command=kb(), meaning searching wikipedia for relevant pages to answer a hard question."
            ),
            cpp_path=RUNTIME_BENCH_DIR / "src" / "approx_kb.c",
            command_parsing_fn=strip_quotes_from_args,
        ),
    ]

    manager = ar.MLIRConfigManager()
    tool_runtimes: Dict[str, ToolRuntime] = {}

    for tool in tools:
        annotated_mlir = compile_cpp_path_to_annotated_mlir(
            tool.cpp_path,
            cgeist_config=cgeist_config,
            toolchain=toolchain,
        )
        params = manager.parse_annotations(annotated_mlir)
        tool_runtimes[tool.name] = ToolRuntime(
            name=tool.name,
            annotated_mlir=annotated_mlir,
            params=params,
            exec_cache={},
            exec_root=Path("/u/haor2/workloads/benchmark/artifacts") / tool.name,
        )
    return tools, tool_runtimes


def _compile_tool_exec(
    tool: ToolRuntime,
    config: Dict[str, int],
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
) -> Path:
    tool.exec_root.mkdir(parents=True, exist_ok=True)
    config_key = _hash_config(config, set(tool.params.keys()))
    if config_key in tool.exec_cache:
        return tool.exec_cache[config_key]

    manager = ar.MLIRConfigManager()
    modified = manager.apply_config(tool.annotated_mlir, config)
    tag = _config_tag(config_key)
    exec_dir = tool.exec_root / tag
    exec_path = compile_mlir_to_native_exec(
        modified,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
        exec_dir=exec_dir,
        tag=tool.name,
    )
    tool.exec_cache[config_key] = exec_path
    return exec_path


def _invoke_tool(
    tool: ToolSpec,
    runtime: ToolRuntime,
    config: Dict[str, int],
    question: str,
    command_str: str,
    necessity_score: int,
    dataset_root: Path,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
) -> Tuple[str, float]:
    exec_path = _compile_tool_exec(runtime, config, cgeist_config, toolchain)
    _LOG.info("tool.compile name=%s exec=%s", tool.name, exec_path)

    args: List[str] = []
    args_str = ""
    match = re.search(r"(\w+)\((.*)\)", command_str)
    if match:
        args_str = match.group(2)
    else:
        _LOG.warning("tool.parse_failed name=%s command=%s", tool.name, command_str)
    if args_str.strip():
        args = next(csv.reader([args_str], skipinitialspace=True))

    stdin = None
    argv: List[str] = []

    if tool.name == "bm25":
        docs_path = dataset_root / "content.txt"
        argv = [str(docs_path), question, str(necessity_score)]
    elif tool.name == "kb":
        question_embedding = _embedding_model.encode(question)
        embedding_str = "[" + ",".join([f"{x: .22f}" for x in question_embedding]) + "]"
        top_k = os.environ.get("KB_TOPK", "5")
        argv = [embedding_str, str(top_k), str(necessity_score)]
        stdin = _read_text(dataset_root / "input.txt")
    else:
        argv = tool.command_parsing_fn(args, runtime)

    _LOG.info("tool.run name=%s argv=%s stdin_bytes=%s", tool.name, argv, 0 if stdin is None else len(stdin))
    result = run_exec(exec_path, argv, stdin=stdin, work_dir=exec_path.parent)
    raw_output = result.stdout.strip()

    if tool.name == "bm25":
        raw_output = rerank(raw_output, question)
    elif tool.name == "kb":
        raw_output = _kb_post_process(raw_output, dataset_root, question)

    _LOG.info("tool.done name=%s time_ms=%.3f out_chars=%d rc=%d", tool.name, result.elapsed_ms, len(raw_output), result.returncode)
    return raw_output, result.elapsed_ms


def _select_tool(llm_output: str, tools: List[ToolSpec]) -> Optional[ToolSpec]:
    # Accept both "tool(...)" and bare "tool" forms.
    match = re.search(r"(\w+)\((.*)\)", llm_output)
    if match:
        tool_name = match.group(1)
    else:
        token = llm_output.strip().split()[0] if llm_output.strip() else ""
        tool_name = token
    return next((t for t in tools if t.name == tool_name), None)


def _extract_command(llm_output: str) -> str:
    # Accept either:
    # 1) command = tool(...)
    # 2) command = none
    # even when "necessity = ..." appears on the same line.
    match = re.search(
        r"command\s*=\s*(none|\w+\([^)\n\r]*\)|\w+)",
        llm_output,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip()


def _tool_prompt(tools: List[ToolSpec], question: str) -> str:
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    return f"""
You have access to the following tools:
{tool_descriptions}

Return EXACTLY two lines and nothing else. Do not include analysis, explanation, reasoning, or extra words.
Line 1 must be a command or none: command=...
Line 2 must be a necessity score (0 to 5) showing how necessary the tool is: necessity=...

If no tool is needed, use:
command=none
necessity=0

Here are some examples:
Question: What are the 10 most important nodes in a graph with 1000 nodes? You must answer as exactly one tuple in the form (x,y). You must floor each coordinate to an integer, for example (12,34) instead of (12.91,34.78).
Return:
command=kmeans(-k,10,-n,1000)
necessity=5

Question: What is the capital of France? You must answer with a short phrase only.
Return:
command=bm25()
necessity=5

Question: Who wrote the novel '1984'? You must answer with a short name or phrase only.
Return:
command=kb()
necessity=5

Question: What is the average energy of a system with boxes1d = 10 after running a molecular dynamics simulation? You must answer with a number only.
Return:
command=lavaMD(-boxes1d,10)
necessity=5

Question: what is 1 + 1? You must answer with a number only.
Return:
command=none
necessity=0

Question: When was Disney's The Fox and the Hound first released? You must answer with a year only.
Return:
command=bm25()
necessity=5

Question: {question}
"""


def _final_answer_prompt(question: str, tool_output: str) -> str:
    return f"""
Original Question: {question}
To help answer your question, you have received the following information from a tool: {tool_output}
Based on tool's output, provide a clear and concise final answer to the original question.
Constraints: 1-2 sentences max, no preamble, no extra explanation, no code, no bullet points.
Final Answer:
"""


def _run_question(
    question: Question,
    tools: List[ToolSpec],
    tool_runtimes: Dict[str, ToolRuntime],
    config: Dict[str, int],
    llm: LLMManager,
    selector_modules,
    variant_names: List[str],
    dataset_root: Path,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
    context_len: int,
) -> Tuple[float, float]:
    global _tokenizer
    _tokenizer = llm.tokenizer

    # Tool selection prompt
    tool_prompt = _tool_prompt(tools, question.text)
    # print(tool_prompt)
    _LOG.info("question.start text=%s", question.text)
    prompt_len = len(_tokenizer.encode(tool_prompt, add_bos=False))
    model_idx = select_variant(selector_modules, prompt_len)
    model_size = variant_names[min(model_idx, len(variant_names) - 1)]
    llm_tool_choice, llm_time_ms = _measure_llm_call(llm, tool_prompt, model_size)
    _LOG.info("llm.tool_choice model=%s time_ms=%.3f output=%s", model_size, llm_time_ms, llm_tool_choice.replace("\n", " "))
    command_str = _extract_command(llm_tool_choice)
    nec_match = re.search(r"necessity\s*=\s*(\d+)", llm_tool_choice, re.IGNORECASE)
    necessity_score = int(nec_match.group(1)) if nec_match else 0
    necessity_score = max(0, min(5, necessity_score))

    selected_tool = _select_tool(command_str, tools)
    tool_output = "No tool was invoked."
    tool_time_ms = 0.0
    if selected_tool:
        tool_output, tool_time_ms = _invoke_tool(
            selected_tool,
            tool_runtimes[selected_tool.name],
            config,
            question.text,
            command_str,
            necessity_score,
            dataset_root,
            cgeist_config,
            toolchain,
        )
    else:
        _LOG.info("tool.skip reason=no_match")
        

    final_prompt = truncate(_final_answer_prompt(question.text, tool_output), context_len, padding=256)
    model_idx = select_variant(selector_modules, int(necessity_score))
    model_size = variant_names[min(model_idx, len(variant_names) - 1)]
    final_answer, final_time_ms = _measure_llm_call(llm, final_prompt, model_size)
    _LOG.info("llm.final model=%s time_ms=%.3f output=%s", model_size, final_time_ms, final_answer.replace("\n", " "))

    accuracy = question.accuracy_fn(final_answer)
    total_time_ms = llm_time_ms + final_time_ms + tool_time_ms
    _LOG.info("question.done acc=%.3f time_ms=%.3f tool_time_ms=%.3f necessity=%d", accuracy, total_time_ms, tool_time_ms, necessity_score)
    return accuracy, total_time_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM tool benchmark with approx-runtime tuning")
    parser.add_argument("--backend", type=str, default=os.environ.get("APPROX_BACKEND", "cuda"))
    parser.add_argument(
        "--llm-backend",
        type=str,
        default=os.environ.get("LLM_BACKEND", "iree"),
        help="LLM sampler backend: iree or xla",
    )
    parser.add_argument("--tuning-time", type=int, default=300)
    parser.add_argument("--accuracy", type=float, default=0.9)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--out-dir", type=str, default="/u/haor2/workloads/benchmark/artifacts")
    parser.add_argument("--variants", type=str, default="1b,4b")
    parser.add_argument("--reference-model", type=str, default="4b")
    parser.add_argument("--use-subset", action="store_true")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--config-log", type=str, default="/u/haor2/workloads/benchmark/artifacts/config_log.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
        force=True,
    )
    _LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    _LOG.propagate = True
    _LOG.info("logging.ready level=%s log_file=%s", args.log_level.upper(), args.log_file)

    dataset_root = LLM_TOOL_DIR
    toolchain = default_toolchain()
    cgeist_config = ar.CgeistConfig(
        cgeist_path=os.environ.get("CGEIST_PATH", _DEFAULT_CGEIST_PATH),
        resource_dir=os.environ.get("CGEIST_RESOURCE_DIR", _DEFAULT_CGEIST_RESOURCE_DIR),
        include_dirs=[os.environ.get("CGEIST_INCLUDE_DIR", _DEFAULT_CGEIST_INCLUDE_DIR)],
    )

    tools, tool_runtimes = _build_tools(toolchain, cgeist_config)

    selector_mlir, selector_pipeline, _ = build_selector_mlir()
    selector_manager = ar.MLIRConfigManager()
    selector_params = selector_manager.parse_annotations(selector_mlir)

    combined_mlir = selector_mlir
    combined_params = dict(selector_params)
    for runtime in tool_runtimes.values():
        combined_mlir += "\n\n" + runtime.annotated_mlir
        for name in runtime.params:
            if name in combined_params:
                raise RuntimeError(f"Parameter name collision across tools: {name}")
            combined_params[name] = runtime.params[name]

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if len(variant_names) < 2:
        raise ValueError("Provide at least two variants, e.g. --variants 1b,4b")

    llm = LLMManager(
        variant_names,
        cache_length=args.context_len,
        backend=args.llm_backend,
    )
    questions = questions_to_run if args.use_subset else evalutation_questions

    extra_args = ["--iree-cuda-target=sm_80", "--iree-cuda-target-features=+ptx76"]
    if args.backend != "cuda":
        extra_args = []

    selector_cache: Dict[Tuple[Tuple[str, int], ...], Tuple[object, Path]] = {}

    def _compile_selector_for_config(config: Dict[str, int]) -> Tuple[object, Path]:
        config_key = _hash_config(config, set(selector_params.keys()))
        if config_key in selector_cache:
            return selector_cache[config_key]
        modified = selector_manager.apply_config(selector_mlir, config)
        modules, vmfb_path = compile_selector(
            modified,
            selector_pipeline,
            args.backend,
            Path(args.out_dir) / "selector",
            extra_args,
        )
        selector_cache[config_key] = (modules, vmfb_path)
        return modules, vmfb_path

    config_log_path = Path(args.config_log)
    config_log_path.parent.mkdir(parents=True, exist_ok=True)
    config_log_exists = config_log_path.exists()

    def _log_config_result(cfg: Dict[str, int], time_ms: float, acc: float) -> None:
        nonlocal config_log_exists
        fieldnames = ["time_ms", "accuracy"] + sorted(combined_params.keys())
        with config_log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not config_log_exists:
                writer.writeheader()
                config_log_exists = True
            row = {"time_ms": f"{time_ms:.6f}", "accuracy": f"{acc:.6f}"}
            row.update(cfg)
            writer.writerow(row)

    def evaluate_fn(config: Dict[str, int]) -> Tuple[float, float]:
        _LOG.info("config.start size=%d", len(config))
        selector_modules, vmfb_path = _compile_selector_for_config(config)
        os.environ["GEMMA_SELECTOR_VMFB"] = str(vmfb_path)

        total_acc = 0.0
        total_time_ms = 0.0
        for idx, question in enumerate(questions, start=1):
            acc, time_ms = _run_question(
                question,
                tools,
                tool_runtimes,
                config,
                llm,
                selector_modules,
                variant_names,
                dataset_root,
                cgeist_config,
                toolchain,
                args.context_len,
            )
            total_acc += acc
            total_time_ms += time_ms
            _LOG.info("progress question=%d/%d acc=%.3f time_ms=%.2f", idx, len(questions), acc, time_ms)

        avg_acc = total_acc / len(questions) if questions else 0.0
        avg_time_ms = total_time_ms / len(questions) if questions else float("inf")
        _log_config_result(config, avg_time_ms, avg_acc) # or, pass it to result_callback
        _LOG.info("config.done time_ms=%.3f acc=%.3f", avg_time_ms, avg_acc)
        return avg_time_ms, avg_acc

    if args.skip_tuning:
        time_ms, acc = evaluate_fn({})
        print(f"Baseline: time={time_ms:.2f}ms, acc={acc:.4f}")
        return

    result = ar.tune(
        mlir_source=combined_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=args.accuracy,
        time_budget=args.tuning_time,
        result_callback=None,
    )

    print("Tuning done.")
    print(f"Best time: {result['best_time']:.2f}ms")
    print(f"Best accuracy: {result['best_accuracy']:.4f}")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
