from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List


@dataclass
class Question:
    """
    Describes a question and its specific function for computing accuracy.
    """

    text: str
    accuracy_fn: Callable[[str], float]


_DEFAULT_NQ_JSON = Path(__file__).resolve().parent / "nq_open_train_qa.json"


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

    tgt_counts: dict[str, int] = {}
    for tok in tgt_tokens:
        tgt_counts[tok] = tgt_counts.get(tok, 0) + 1

    overlap = 0
    for tok in pred_tokens:
        count = tgt_counts.get(tok, 0)
        if count > 0:
            overlap += 1
            tgt_counts[tok] = count - 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    return 2 * precision * recall / (precision + recall)


def acc_nq_short_answer(predicted_answer: str, expected_answers: List[str]) -> float:
    """
    Score a model answer against NQ short answers.
    Returns max score over aliases:
    - 1.0 for normalized exact/substring match
    - otherwise token-level F1 in [0, 1]
    """
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


def load_nq_qa_pairs(json_path: Path | str = _DEFAULT_NQ_JSON) -> list[dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def build_nq_questions(qa_pairs: list[dict]) -> list[Question]:
    questions: list[Question] = []
    for item in qa_pairs:
        text = item.get("question")
        answers = item.get("answers", [])
        if not isinstance(text, str):
            continue
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            continue
        questions.append(
            Question(
                text=text,
                accuracy_fn=lambda pred, expected=answers: acc_nq_short_answer(pred, expected),
            )
        )
    return questions


def load_nq_questions(json_path: Path | str = _DEFAULT_NQ_JSON) -> list[Question]:
    return build_nq_questions(load_nq_qa_pairs(json_path))


evalutation_questions = load_nq_questions()
questions_to_run = evalutation_questions
