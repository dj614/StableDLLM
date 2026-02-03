from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .extract import extract_boxed_answer, extract_gsm8k_hash_answer, extract_last_number


def _normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s


def _loose_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    return _normalize(pred) == _normalize(gold)


@dataclass
class ScoreResult:
    total: int
    correct: int
    extraction_rate: float

    def to_dict(self) -> Dict[str, Any]:
        acc = self.correct / self.total if self.total else 0.0
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": acc,
            "extraction_rate": self.extraction_rate,
        }


def score_gsm8k(rows: Iterable[Dict[str, Any]]) -> ScoreResult:
    total = 0
    correct = 0
    extracted = 0
    for r in rows:
        total += 1
        pred_text = r.get("prediction") or r.get("pred_text") or ""
        gold_raw = r.get("gold_raw") or r.get("answer") or r.get("gold") or r.get("output") or ""
        pred = extract_gsm8k_hash_answer(pred_text) or extract_last_number(pred_text)
        gold = extract_gsm8k_hash_answer(gold_raw) or extract_last_number(gold_raw)
        if pred is not None:
            extracted += 1
        if _loose_match(pred, gold):
            correct += 1
    extraction_rate = extracted / total if total else 0.0
    return ScoreResult(total=total, correct=correct, extraction_rate=extraction_rate)


def score_openscience(rows: Iterable[Dict[str, Any]]) -> ScoreResult:
    total = 0
    correct = 0
    extracted = 0
    for r in rows:
        total += 1
        pred_text = r.get("prediction") or ""
        gold_raw = r.get("gold_raw") or r.get("output") or r.get("answer") or ""
        pred = extract_boxed_answer(pred_text) or extract_last_number(pred_text)
        gold = extract_boxed_answer(gold_raw) or extract_last_number(gold_raw)
        if pred is not None:
            extracted += 1
        if _loose_match(pred, gold):
            correct += 1
    extraction_rate = extracted / total if total else 0.0
    return ScoreResult(total=total, correct=correct, extraction_rate=extraction_rate)


def score_hitab(rows: Iterable[Dict[str, Any]]) -> ScoreResult:
    """HiTab datasets vary; we attempt a few common gold fields."""
    total = 0
    correct = 0
    extracted = 0
    for r in rows:
        total += 1
        pred_text = r.get("prediction") or ""
        gold_raw = (
            r.get("gold_raw")
            or r.get("ground_truth")
            or r.get("answer")
            or r.get("label")
            or ""
        )
        pred = extract_boxed_answer(pred_text) or extract_gsm8k_hash_answer(pred_text) or extract_last_number(pred_text)
        gold = extract_boxed_answer(gold_raw) or extract_gsm8k_hash_answer(gold_raw) or extract_last_number(gold_raw)
        if pred is not None:
            extracted += 1
        if _loose_match(pred, gold):
            correct += 1
    extraction_rate = extracted / total if total else 0.0
    return ScoreResult(total=total, correct=correct, extraction_rate=extraction_rate)
