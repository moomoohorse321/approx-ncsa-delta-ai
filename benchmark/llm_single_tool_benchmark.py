#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import approx_runtime as ar
import jax
import jax.numpy as jnp
from gemma.gm.iree import _iree as gemma_iree

from .llm import LLMManager
from . import tool_runtime_common as trc

_LOG = logging.getLogger("llm_single_tool_bench")

_DEFAULT_CGEIST_PATH = "/work/nvme/beqg/PolygeistSample/build/bin/cgeist"
_DEFAULT_CGEIST_RESOURCE_DIR = "/work/nvme/beqg/PolygeistSample/llvm-project/build/lib/clang/18"
_DEFAULT_CGEIST_INCLUDE_DIR = "/work/nvme/beqg/PolygeistSample/tools/cgeist/Test/polybench/utilities"

_ACCURACY_FLOOR_THRESHOLD = 0.2
_MODEL_CONTEXT_LENGTHS = {
    "1b": 1024,
    "4b": 16384,
}
_SIMILARITY_TOOL_OUTPUT_MAX_TOKENS = 4096

os.environ.setdefault("CGEIST_PATH", _DEFAULT_CGEIST_PATH)
os.environ.setdefault("CGEIST_RESOURCE_DIR", _DEFAULT_CGEIST_RESOURCE_DIR)
os.environ.setdefault("CGEIST_INCLUDE_DIR", _DEFAULT_CGEIST_INCLUDE_DIR)

RUNTIME_BENCH_DIR = Path(
    os.environ.get(
        "APPROX_RUNTIME_BENCH_DIR",
        "/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime/examples/benchmark",
    )
)
if str(RUNTIME_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_BENCH_DIR))
from benchmark_common import default_toolchain, exact_decision_config


def reset_gemma_stats() -> None:
    if hasattr(gemma_iree, "_STATS"):
        gemma_iree._STATS.update({"total_ms": 0.0, "transfer_ms": 0.0, "post_ms": 0.0, "count": 0})


def get_total_compute_ms() -> float:
    stats = getattr(gemma_iree, "_STATS", None)
    if not stats or stats.get("count", 0) == 0:
        raise RuntimeError("No Gemma IREE stats recorded for total_compute")
    return stats["total_ms"] - stats["transfer_ms"] - stats["post_ms"]


def measure_llm_call(llm, prompt: str, model_size: str) -> Tuple[str, float]:
    if getattr(llm, "backend", "iree") == "iree":
        reset_gemma_stats()
        output = llm.generate(prompt, model_size)
        return output, get_total_compute_ms()
    start = time.perf_counter()
    output = llm.generate(prompt, model_size)
    return output, (time.perf_counter() - start) * 1000.0


def truncate(tokenizer, text: str, context_len: int, padding: int = 128, reserved: int = 64) -> str:
    max_len = context_len - padding - reserved
    token_ids_list = tokenizer.encode(text, add_bos=False)
    if len(token_ids_list) >= max_len:
        token_ids_list = token_ids_list[: -padding][: max_len - reserved] + token_ids_list[-padding:]
    return tokenizer.decode(token_ids_list)


def _first_n_tokens(tokenizer, text: str, max_tokens: int) -> str:
    token_ids_list = tokenizer.encode(text, add_bos=False)
    if len(token_ids_list) <= max_tokens:
        return text
    return tokenizer.decode(token_ids_list[:max_tokens])


def final_answer_prompt(question: str, tool_output: str) -> str:
    return f"""
Original Question: {question}
To help answer your question, you have received the following information from a tool: {tool_output}
Give the final answer in a short phrase.
Rules:
- 1 to 6 words only
- no full sentence
- no explanation
- no extra text

Examples:
Q: who wrote the novel 1984
A: George Orwell

Q: when was disney the fox and the hound first released
A: 1981

Final Answer:
"""


def _normalize_question_text(text: str) -> str:
    text = text.lower().strip().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_nq_generated_questions(
    sample_size: int | None = None,
    seed: int = 0,
    tuning_questions_in_order: list[str] | None = None,
):
    from .questions_nq_generated import evalutation_questions

    if sample_size is not None:
        if sample_size <= 0:
            return []
        if sample_size >= len(evalutation_questions):
            return list(evalutation_questions)
        rng = random.Random(seed)
        selected_questions = rng.sample(list(evalutation_questions), sample_size)
        _LOG.info(
            "using sampled question list for single-tool benchmark: sample_size=%d seed=%d",
            sample_size,
            seed,
        )
        return selected_questions

    if tuning_questions_in_order is None:
        return list(evalutation_questions)

    question_by_key = {}
    for question in evalutation_questions:
        key = _normalize_question_text(question.text)
        if key not in question_by_key:
            question_by_key[key] = question

    selected_questions = []
    missing_questions = []
    for question_text in tuning_questions_in_order:
        key = _normalize_question_text(question_text)
        question = question_by_key.get(key)
        if question is None:
            missing_questions.append(question_text)
            continue
        selected_questions.append(question)

    if missing_questions:
        missing = "; ".join(missing_questions)
        raise RuntimeError(f"Missing requested questions from evaluation set: {missing}")

    _LOG.info(
        "using fixed question list for single-tool benchmark: requested=%d selected=%d",
        len(tuning_questions_in_order),
        len(selected_questions),
    )
    return selected_questions


def get_state(similarity_score_x10: jax.Array) -> jax.Array:
    return similarity_score_x10


def _build_static_knob_plan(param_names: set[str]) -> Tuple[Dict[str, int], set[str], set[str]]:
    fixed_config: Dict[str, int] = {}
    decision_alias_bases: set[str] = set()
    threshold_bases: set[str] = set()

    for name in param_names:
        if name.endswith("_decision_1"):
            base = name[: -len("_decision_1")]
            fixed_config[name] = 0
            decision_alias_bases.add(base)
        elif name.endswith("_threshold_0"):
            base = name[: -len("_threshold_0")]
            fixed_config[name] = 0
            threshold_bases.add(base)

    return fixed_config, decision_alias_bases, threshold_bases


def _expand_static_config(config: Dict[str, int], decision_alias_bases: set[str], threshold_bases: set[str]) -> Dict[str, int]:
    expanded = dict(config)
    for base in decision_alias_bases:
        d0 = f"{base}_decision_0"
        d1 = f"{base}_decision_1"
        if d0 in expanded:
            expanded[d1] = expanded[d0]
    for base in threshold_bases:
        expanded[f"{base}_threshold_0"] = 0
    return expanded


def _load_replay_configs(csv_path: Path, allowed_keys: set[str]) -> List[Dict[str, int]]:
    configs: List[Dict[str, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg: Dict[str, int] = {}
            for key, raw in row.items():
                if key not in allowed_keys:
                    continue
                if raw is None or raw == "":
                    continue
                try:
                    cfg[key] = int(float(raw))
                except ValueError:
                    continue
            configs.append(cfg)
    return configs


@ar.Knob(
    decision_tree=ar.DecisionTree(
        state_function=get_state,
        state_indices=[0],
        thresholds=[5],
        decisions=[0, 0],
        transform_type="task_skipping",
        thresholds_lower=[0],
        thresholds_upper=[10],
        decision_values=[0, 1],
    )
)
def selector_kernel(similarity_score_x10: jax.Array) -> jax.Array:
    return jax.lax.cond(
        similarity_score_x10 >= 5,
        lambda _: jnp.int32(0),
        lambda _: jnp.int32(1),
        operand=None,
    )


def build_selector_mlir() -> Tuple[str, list[str], Dict]:
    config = ar.get_config(selector_kernel)
    shape = jax.ShapeDtypeStruct((), jnp.int32)
    functions = {
        "selector_kernel": (selector_kernel, (shape,), config),
        "get_state": (get_state, (shape,), None),
    }
    mlir = ar.export_module_with_configs(functions)
    pipeline = ar.get_pipeline_for_config(config)
    return mlir, pipeline, config


def compile_selector(
    mlir_annotated: str,
    pipeline: list[str],
    backend: str,
    out_dir: Path,
    extra_args: list[str],
):
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


def select_variant(modules, similarity_score: float) -> int:
    result = modules.module["selector_kernel"](jnp.array(similarity_score, dtype=jnp.int32))
    if hasattr(result, "to_host"):
        result = result.to_host()
    return int(result)


def _run_single_question(
    question,
    llm: LLMManager,
    variant_names,
    selector_modules,
    tool,
    runtime,
    config: Dict[str, int],
    dataset_root: Path,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
    context_len: int,
    necessity_score: int,
    model_context_lengths: Dict[str, int],
    sim_score_min: float,
    sim_score_max: float,
) -> Tuple[float, float]:
    command_str = f"{tool.name}()"
    tool_output, tool_time_ms = trc.invoke_tool(
        tool,
        runtime,
        config,
        question.text,
        command_str,
        necessity_score,
        dataset_root,
        cgeist_config,
        toolchain,
        logger=_LOG,
    )
    sim_input = _first_n_tokens(llm.tokenizer, tool_output, _SIMILARITY_TOOL_OUTPUT_MAX_TOKENS)
    sim_score = trc.similarity_score(sim_input, question.text)
    sim_score_clipped = min(sim_score_max, max(sim_score_min, sim_score))
    if sim_score_max > sim_score_min:
        sim_score_norm = (sim_score_clipped - sim_score_min) / (sim_score_max - sim_score_min)
    else:
        sim_score_norm = 0.0
    sim_score_x10 = max(0, min(10, int(round(sim_score_norm * 10.0))))
    idx = select_variant(selector_modules, sim_score_x10)
    if idx < 0 or idx >= len(variant_names):
        idx = 0
    model_size = variant_names[idx]
    _LOG.info(
        "llm.select model=%s sim_score=%.4f sim_score_x10=%d idx=%d",
        model_size,
        sim_score,
        sim_score_x10,
        idx,
    )
    selected_context_len = model_context_lengths.get(model_size, context_len)
    final_prompt = truncate(
        llm.tokenizer,
        final_answer_prompt(question.text, tool_output),
        selected_context_len,
        padding=256,
    )
    final_answer, llm_time_ms = measure_llm_call(llm, final_prompt, model_size)
    _LOG.info("llm.infer.done model=%s llm_time_ms=%.3f", model_size, llm_time_ms)
    _LOG.info("llm.infer.answer model=%s text=%s", model_size, final_answer)
    acc = question.accuracy_fn(final_answer)
    if acc < _ACCURACY_FLOOR_THRESHOLD:
        acc = 0.0

    return acc, llm_time_ms + tool_time_ms


def run_single_tool_benchmark(
    tool_name: str,
    tuning_questions_in_order: list[str] | None = None,
    sim_score_min: float = 0.0,
    sim_score_max: float = 1.0,
) -> None:
    parser = argparse.ArgumentParser(description=f"LLM {tool_name} benchmark with approx-runtime tuning")
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
    parser.add_argument("--out-dir", type=str, default=f"/u/haor2/workloads/benchmark/artifacts/{tool_name}")
    parser.add_argument("--variants", type=str, default="4b,1b")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--necessity-score", type=int, default=5)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--config-log", type=str, default="")
    parser.add_argument("--database", type=str, default="opentuner.db")
    parser.add_argument("--static", action="store_true", help="Collapse all branch knobs into static knobs.")
    parser.add_argument("--exact", action="store_true", help="Run explicit exact baseline mode with tuning question slice.")
    parser.add_argument("--replay-config-csv", type=str, default="", help="Replay configs from this CSV instead of tuning.")
    parser.add_argument("--replay-output-csv", type=str, default="", help="Output CSV path for replay results.")
    parser.add_argument(
        "--replay-split",
        type=str,
        default="",
        choices=["", "eval", "tune"],
        help="Question split to use during replay (eval or tune).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(args.log_file)] if args.log_file else []),
        force=True,
    )
    _LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    _LOG.propagate = True

    if args.replay_config_csv:
        args.skip_tuning = True

    necessity_score = max(0, min(5, int(args.necessity_score)))
    dataset_root = Path(__file__).resolve().parent
    toolchain = default_toolchain()
    cgeist_config = ar.CgeistConfig(
        cgeist_path=os.environ.get("CGEIST_PATH", _DEFAULT_CGEIST_PATH),
        resource_dir=os.environ.get("CGEIST_RESOURCE_DIR", _DEFAULT_CGEIST_RESOURCE_DIR),
        include_dirs=[os.environ.get("CGEIST_INCLUDE_DIR", _DEFAULT_CGEIST_INCLUDE_DIR)],
    )

    tools, tool_runtimes = trc.build_tools(
        toolchain,
        cgeist_config,
        exec_root_base=Path(args.out_dir),
        tool_filter=[tool_name],
    )
    if len(tools) != 1:
        raise RuntimeError(f"Failed to initialize tool={tool_name}")
    tool = tools[0]
    runtime = tool_runtimes[tool_name]

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if len(variant_names) < 2:
        raise ValueError("Provide at least two variants, e.g. --variants 1b,4b")

    model_context_lengths = {v: _MODEL_CONTEXT_LENGTHS.get(v, args.context_len) for v in variant_names}
    llm = LLMManager(
        variant_names,
        cache_length=model_context_lengths,
        backend=args.llm_backend,
    )
    questions = get_nq_generated_questions(args.sample_size, args.sample_seed, tuning_questions_in_order)
    if not questions:
        raise RuntimeError("No questions loaded from questions_nq_generated.py")

    selector_mlir, selector_pipeline, _ = build_selector_mlir()
    selector_manager = ar.MLIRConfigManager()
    selector_params = selector_manager.parse_annotations(selector_mlir)
    selector_cache: Dict[Tuple[Tuple[str, int], ...], Tuple[object, Path]] = {}
    combined_mlir = selector_mlir + "\n\n" + runtime.annotated_mlir
    tune_mlir = combined_mlir
    decision_alias_bases: set[str] = set()
    threshold_bases: set[str] = set()
    if args.static:
        all_param_names = set(runtime.params.keys()) | set(selector_params.keys())
        static_fixed_config, decision_alias_bases, threshold_bases = _build_static_knob_plan(all_param_names)
        if static_fixed_config:
            combined_manager = ar.MLIRConfigManager()
            tune_mlir = combined_manager.apply_config(combined_mlir, static_fixed_config)
        _LOG.info(
            "static mode enabled: fixed_params=%d alias_groups=%d threshold_groups=%d",
            len(static_fixed_config),
            len(decision_alias_bases),
            len(threshold_bases),
        )

    extra_args = ["--iree-cuda-target=sm_80", "--iree-cuda-target-features=+ptx76"]
    if args.backend != "cuda":
        extra_args = []

    config_log_path = args.config_log or str(Path(args.out_dir) / "config_log.csv")
    config_log = Path(config_log_path)
    config_log.parent.mkdir(parents=True, exist_ok=True)
    config_log_exists = config_log.exists()

    def _log_config_result(cfg: Dict[str, int], time_ms: float, acc: float) -> None:
        nonlocal config_log_exists
        fieldnames = ["time_ms", "accuracy"] + sorted(set(runtime.params.keys()) | set(selector_params.keys()))
        with config_log.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not config_log_exists:
                writer.writeheader()
                config_log_exists = True
            row = {"time_ms": f"{time_ms:.6f}", "accuracy": f"{acc:.6f}"}
            row.update(cfg)
            writer.writerow(row)

    def evaluate_fn(config: Dict[str, int]) -> Tuple[float, float]:
        effective_config = _expand_static_config(config, decision_alias_bases, threshold_bases) if args.static else config
        _LOG.info("config.start size=%d effective_size=%d", len(config), len(effective_config))
        selector_key = tuple(sorted((k, v) for k, v in effective_config.items() if k in selector_params))
        if selector_key in selector_cache:
            selector_modules, vmfb_path = selector_cache[selector_key]
        else:
            modified_selector = selector_manager.apply_config(selector_mlir, effective_config)
            selector_modules, vmfb_path = compile_selector(
                modified_selector,
                selector_pipeline,
                args.backend,
                Path(args.out_dir) / "selector",
                extra_args,
            )
            selector_cache[selector_key] = (selector_modules, vmfb_path)
        os.environ["GEMMA_SELECTOR_VMFB"] = str(vmfb_path)

        total_acc = 0.0
        total_time_ms = 0.0
        for idx, question in enumerate(questions, start=1):
            acc, time_ms = _run_single_question(
                question,
                llm,
                variant_names,
                selector_modules,
                tool,
                runtime,
                effective_config,
                dataset_root,
                cgeist_config,
                toolchain,
                args.context_len,
                necessity_score,
                model_context_lengths,
                sim_score_min,
                sim_score_max,
            )
            total_acc += acc
            total_time_ms += time_ms
            _LOG.info("progress question=%d/%d acc=%.3f time_ms=%.2f", idx, len(questions), acc, time_ms)

        avg_acc = total_acc / len(questions)
        avg_time_ms = total_time_ms / len(questions)
        _LOG.info("config.done time_ms=%.3f acc=%.3f", avg_time_ms, avg_acc)
        return avg_time_ms, avg_acc

    def result_callback(config: Dict[str, int], time_ms: float, accuracy: float) -> None:
        effective_config = _expand_static_config(config, decision_alias_bases, threshold_bases) if args.static else config
        _log_config_result(effective_config, time_ms, accuracy)

    if args.replay_config_csv:
        replay_csv_path = Path(args.replay_config_csv)
        replay_output_path = Path(args.replay_output_csv) if args.replay_output_csv else Path(args.out_dir) / "replay_results.csv"
        replay_output_path.parent.mkdir(parents=True, exist_ok=True)

        allowed_keys = set(runtime.params.keys()) | set(selector_params.keys())
        replay_configs = _load_replay_configs(replay_csv_path, allowed_keys)
        if not replay_configs:
            raise RuntimeError(f"No replay configs loaded from {replay_csv_path}")

        result_fieldnames = ["idx", "time_ms", "accuracy"] + sorted(allowed_keys)
        with replay_output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_fieldnames)
            writer.writeheader()
            for idx, cfg in enumerate(replay_configs, start=1):
                time_ms, acc = evaluate_fn(cfg)
                effective_config = _expand_static_config(cfg, decision_alias_bases, threshold_bases) if args.static else cfg
                row: Dict[str, Any] = {"idx": idx, "time_ms": f"{time_ms:.6f}", "accuracy": f"{acc:.6f}"}
                row.update(effective_config)
                writer.writerow(row)
                _LOG.info("replay progress=%d/%d time_ms=%.3f acc=%.3f", idx, len(replay_configs), time_ms, acc)
        print(f"Replay done. rows={len(replay_configs)} output={replay_output_path}")
        return

    if args.skip_tuning:
        all_params: Dict[str, ar.TunableParam] = {}
        all_params.update(runtime.params)
        all_params.update(selector_params)
        baseline_config = exact_decision_config(all_params)
        time_ms, acc = evaluate_fn(baseline_config)
        result_callback(baseline_config, time_ms, acc)
        print(f"Baseline ({tool_name}): time={time_ms:.2f}ms, acc={acc:.4f}")
        return

    result = ar.tune(
        mlir_source=tune_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=args.accuracy,
        time_budget=args.tuning_time,
        database=args.database,
        result_callback=result_callback,
    )
    best_config = _expand_static_config(result["best_config"], decision_alias_bases, threshold_bases) if args.static else result["best_config"]
    print("Tuning done.")
    print(f"Best time: {result['best_time']:.2f}ms")
    print(f"Best accuracy: {result['best_accuracy']:.4f}")
    print(f"Best config: {best_config}")
