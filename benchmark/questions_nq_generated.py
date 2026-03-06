from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Question:
    """
    Describes a question and its specific function for computing accuracy.
    """
    text: str
    accuracy_fn: Callable[[str], float]


_NQ_JSON_PATH = Path(r"/u/haor2/workloads/benchmark/nq_open_train_qa.json")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(prediction: str, target: str) -> float:
    pred_tokens = prediction.split()
    tgt_tokens = target.split()
    if not pred_tokens or not tgt_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for t in tgt_tokens:
        counts[t] = counts.get(t, 0) + 1
    overlap = 0
    for t in pred_tokens:
        c = counts.get(t, 0)
        if c > 0:
            overlap += 1
            counts[t] = c - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    return 2 * precision * recall / (precision + recall)


def acc_nq_short_answer(predicted_answer: str, expected_answers: list[str]) -> float:
    pred = _normalize_text(predicted_answer)
    if not pred:
        return 0.0
    best = 0.0
    for ans in expected_answers:
        tgt = _normalize_text(ans)
        if not tgt:
            continue
        if pred == tgt or tgt in pred or pred in tgt:
            return 1.0
        best = max(best, _token_f1(pred, tgt))
    return best


def _load_pairs() -> list[dict]:
    with _NQ_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {_NQ_JSON_PATH}")
    return data


def _build_questions() -> list[Question]:
    out: list[Question] = []
    for item in _load_pairs():
        q = item.get("question")
        a = item.get("answers", [])
        if isinstance(q, str) and isinstance(a, list) and all(isinstance(x, str) for x in a):
            out.append(Question(text=q, accuracy_fn=lambda pred, exp=a: acc_nq_short_answer(pred, exp)))
    return out


evalutation_questions = _build_questions()