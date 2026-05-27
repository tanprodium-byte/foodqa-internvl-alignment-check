#!/usr/bin/env python3
"""Inspect FoodQA prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foodqa.metrics import summarize_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Prediction JSONL path.")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to print.")
    args = parser.parse_args()

    metrics = summarize_predictions(args.pred)
    print("=== Metrics Summary ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("=== Predictions ===")
    shown = 0
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            print("-" * 80)
            print(f"ID: {record.get('id')}")
            print(f"row_index: {record.get('row_index')}")
            print(f"image_id: {record.get('image_id')}")
            print(f"type: {record.get('question_type')}")
            print(f"question: {record.get('question')}")
            print(f"GT: {record.get('gt_answer')}")
            print(f"PRED: {record.get('pred_answer')}")
            print(f"error: {record.get('error')}")
            print(f"used_max_dynamic_patch: {record.get('used_max_dynamic_patch')}")
            print(f"fallback_used: {record.get('fallback_used')}")
            print(f"elapsed_sec: {record.get('elapsed_sec')}")
            shown += 1
            if args.limit is not None and shown >= args.limit:
                break


if __name__ == "__main__":
    main()
