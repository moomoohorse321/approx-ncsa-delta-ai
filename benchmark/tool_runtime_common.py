#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import approx_runtime as ar
import torch
from sentence_transformers import SentenceTransformer, util

_LOG = logging.getLogger("tool_runtime_common")

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


_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")


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


def _tool_catalog() -> List[ToolSpec]:
    return [
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


def build_tools(
    toolchain: ar.ToolchainConfig,
    cgeist_config: ar.CgeistConfig,
    exec_root_base: Path = Path("/u/haor2/workloads/benchmark/artifacts"),
    tool_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[ToolSpec], Dict[str, ToolRuntime]]:
    tools = _tool_catalog()
    if tool_filter:
        selected = set(tool_filter)
        tools = [t for t in tools if t.name in selected]

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
            exec_root=exec_root_base / tool.name,
        )
    return tools, tool_runtimes


def compile_tool_exec(
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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


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


def similarity_score(docs_string: str, query: str) -> float:
    docs = [line for line in docs_string.strip().split("\n") if line.strip()]
    if not docs:
        return 0.0
    query_embedding = _embedding_model.encode(query, convert_to_tensor=True, device=torch.device("cuda"))
    doc_embeddings = _embedding_model.encode(docs, convert_to_tensor=True, device=torch.device("cuda"))
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    return float(torch.max(scores).item())


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


def _parse_tool_time_ms(tool_name: str, stdout: str) -> float:
    if tool_name == "bm25":
        match = re.search(r"\bComputation time:\s*([\d.]+)\s*ms\b", stdout)
    elif tool_name == "kb":
        match = re.search(r"\bElapsed\s+([\d.]+)\s*ms\b", stdout)
    else:
        match = re.search(r"\b(?:Computation time:|Elapsed)\s*([\d.]+)\s*ms\b", stdout)

    if not match:
        raise RuntimeError(f"{tool_name}: failed to parse execution time from stdout")
    return float(match.group(1))


def invoke_tool(
    tool: ToolSpec,
    runtime: ToolRuntime,
    config: Dict[str, int],
    question: str,
    command_str: str,
    necessity_score: int,
    dataset_root: Path,
    cgeist_config: ar.CgeistConfig,
    toolchain: ar.ToolchainConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, float]:
    log = logger or _LOG
    exec_path = compile_tool_exec(runtime, config, cgeist_config, toolchain)
    log.info("tool.compile name=%s exec=%s", tool.name, exec_path)

    args: List[str] = []
    args_str = ""
    match = re.search(r"(\w+)\((.*)\)", command_str)
    if match:
        args_str = match.group(2)
    else:
        log.warning("tool.parse_failed name=%s command=%s", tool.name, command_str)
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

    log.info("tool.run name=%s argv=%s stdin_bytes=%s", tool.name, argv, 0 if stdin is None else len(stdin))
    result = run_exec(exec_path, argv, stdin=stdin, work_dir=exec_path.parent)
    tool_time_ms = _parse_tool_time_ms(tool.name, result.stdout)
    raw_output = result.stdout.strip()

    if tool.name == "bm25":
        raw_output = rerank(raw_output, question)
    elif tool.name == "kb":
        raw_output = _kb_post_process(raw_output, dataset_root, question)

    log.info(
        "tool.done name=%s time_ms=%.3f out_chars=%d rc=%d",
        tool.name,
        tool_time_ms,
        len(raw_output),
        result.returncode,
    )
    return raw_output, tool_time_ms
