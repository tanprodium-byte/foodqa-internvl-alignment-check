"""Simple deterministic metrics for FoodQA JSONL predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .prompts import is_mcq


def extract_mcq_letter(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None

    patterns = [
        r"(?i)\bđáp\s*án\s*[:：]?\s*([ABCD])\b",
        r"(?i)\banswer\s*[:：]?\s*([ABCD])\b",
        r"^\s*([ABCD])\b",
        r"\b([ABCD])\s*[\.\)\-:]",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1).upper()
    return None


def summarize_predictions(jsonl_path: str | Path) -> dict[str, Any]:
    total = 0
    errors = 0
    non_null_predictions = 0
    mcq_count = 0
    mcq_eval_count = 0
    mcq_correct = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1

            if record.get("error"):
                errors += 1
            if record.get("pred_answer"):
                non_null_predictions += 1

            if is_mcq(record.get("question_type"), record.get("question")):
                mcq_count += 1
                gt_letter = _extract_gt_letter(record.get("gt_answer"))
                pred_letter = extract_mcq_letter(record.get("pred_answer"))
                if gt_letter is not None:
                    mcq_eval_count += 1
                    if pred_letter == gt_letter:
                        mcq_correct += 1

    accuracy = mcq_correct / mcq_eval_count if mcq_eval_count else None
    return {
        "total_records": total,
        "records_with_error": errors,
        "non_null_predictions": non_null_predictions,
        "mcq_count": mcq_count,
        "mcq_ground_truth_letter_count": mcq_eval_count,
        "mcq_exact_letter_correct": mcq_correct,
        "mcq_exact_letter_accuracy": accuracy,
    }


def _extract_gt_letter(text: str | None) -> str | None:
    if text is None:
        return None
    match = re.fullmatch(r"\s*([ABCD])\s*", str(text).strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None
