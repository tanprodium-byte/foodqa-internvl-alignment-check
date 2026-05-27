#!/usr/bin/env python3
"""Run zero-shot InternVL inference on the matched FoodQA CSV."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foodqa.hf_utils import check_hf_auth, upload_json_to_hf, upload_prediction_file_to_hf  # noqa: E402
from foodqa.internvl_runner import load_config, run_zero_shot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--output", default=None, help="Override output JSONL path.")
    parser.add_argument("--max-dynamic-patch", type=int, default=None, help="Override max dynamic image patches.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max generated tokens.")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed id/row_index records.")
    parser.add_argument(
        "--sample-distinct-images",
        type=int,
        default=None,
        metavar="N",
        help="Sample up to N rows with distinct image_id values before inference.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for --sample-distinct-images.")
    parser.add_argument("--hf-upload", action="store_true", help="Upload latest/final prediction artifacts to HF.")
    parser.add_argument("--hf-repo-id", default=None, help="Hugging Face repo id, e.g. USER_OR_ORG/DATASET_REPO.")
    parser.add_argument("--hf-repo-type", default=None, help="Hugging Face repo type. Default: dataset.")
    parser.add_argument("--hf-upload-every-records", type=int, default=None, help="Periodic HF upload record interval.")
    parser.add_argument("--hf-upload-every-minutes", type=float, default=None, help="Periodic HF upload minute interval.")
    parser.add_argument("--hf-keep-checkpoints", action="store_true", help="Also upload periodic step checkpoint files.")
    parser.add_argument("--hf-prune-remote", action="store_true", help="Prune old visible remote prediction files.")
    parser.add_argument("--run-name", default=None, help="Run name used for HF prediction artifact paths.")
    parser.add_argument("--hf-dry-run", action="store_true", help="Print HF upload paths without inference or network calls.")
    parser.add_argument("--hf-test-upload", action="store_true", help="Upload tiny test JSON artifacts, then exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output or config["output_path"])
    run_name = args.run_name or config.get("run_name") or output_path.stem
    hf_repo_id = args.hf_repo_id or config.get("hf_repo_id")
    hf_repo_type = args.hf_repo_type or config.get("hf_repo_type", "dataset")
    hf_keep_checkpoints = args.hf_keep_checkpoints or bool(config.get("hf_keep_checkpoints", False))
    hf_prune_remote = args.hf_prune_remote or bool(config.get("hf_prune_remote", False))

    if args.hf_dry_run:
        print("=== HF Dry Run ===")
        print(f"repo_id: {hf_repo_id}")
        print(f"repo_type: {hf_repo_type}")
        print("would upload:")
        print(f"predictions/{run_name}.latest.jsonl")
        print(f"predictions/{run_name}.summary.json")
        print(f"predictions/{run_name}.final.jsonl")
        print(f"checkpoints kept: {hf_keep_checkpoints}")
        print(f"remote prune requested: {hf_prune_remote}")
        return

    if args.hf_test_upload:
        _run_hf_test_upload(repo_id=hf_repo_id, repo_type=hf_repo_type)
        return

    if args.hf_upload and hf_repo_id and _looks_like_placeholder_repo_id(hf_repo_id):
        raise SystemExit(f"Refusing HF upload to placeholder repo id: {hf_repo_id}")

    data_csv = Path(config.get("matched_csv") or config["data_csv"])
    if not data_csv.exists():
        data_csv = Path(config["data_csv"])

    run_zero_shot(
        data_csv=data_csv,
        image_root=config["image_root"],
        output_path=output_path,
        model_name=config.get("model_name", "OpenGVLab/InternVL3_5-2B-Instruct"),
        limit=args.limit,
        input_size=int(config.get("input_size", 448)),
        max_dynamic_patch=(
            args.max_dynamic_patch
            if args.max_dynamic_patch is not None
            else int(config.get("max_dynamic_patch", 2))
        ),
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else int(config.get("max_new_tokens", 128))
        ),
        torch_dtype=config.get("torch_dtype", "float16"),
        disable_cudnn=bool(config.get("disable_cudnn", True)),
        use_flash_attn=bool(config.get("use_flash_attn", False)),
        batch_size=int(config.get("batch_size", 1)),
        resume=args.resume or bool(config.get("resume", False)),
        sample_distinct_images=args.sample_distinct_images,
        seed=args.seed,
        hf_upload=args.hf_upload,
        hf_repo_id=hf_repo_id,
        hf_repo_type=hf_repo_type,
        hf_upload_every_records=(
            args.hf_upload_every_records
            if args.hf_upload_every_records is not None
            else int(config.get("hf_upload_every_records", 1000))
        ),
        hf_upload_every_minutes=(
            args.hf_upload_every_minutes
            if args.hf_upload_every_minutes is not None
            else float(config.get("hf_upload_every_minutes", 30.0))
        ),
        hf_keep_checkpoints=hf_keep_checkpoints,
        hf_prune_remote=hf_prune_remote,
        run_name=run_name,
    )


def _run_hf_test_upload(repo_id: str | None, repo_type: str) -> None:
    if not repo_id:
        raise SystemExit("--hf-test-upload requires --hf-repo-id.")
    if _looks_like_placeholder_repo_id(repo_id):
        raise SystemExit(f"Refusing HF test upload to placeholder repo id: {repo_id}")

    ok, message = check_hf_auth(repo_id=repo_id, repo_type=repo_type)
    if not ok:
        raise SystemExit(f"HF auth/access check failed: {message}")
    print(f"HF auth: {message}")

    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    local_path = outputs_dir / "hf_upload_test.json"
    payload = {
        "ok": True,
        "purpose": "foodqa hf upload smoke test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    ok, message = upload_prediction_file_to_hf(
        local_path=local_path,
        repo_id=repo_id,
        path_in_repo="predictions/hf_upload_test.json",
        repo_type=repo_type,
        commit_message="FoodQA HF upload smoke test",
    )
    if not ok:
        raise SystemExit(message)
    print(message)

    ok, message = upload_json_to_hf(
        payload={
            "test_file": "predictions/hf_upload_test.json",
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        repo_id=repo_id,
        path_in_repo="predictions/hf_upload_test.summary.json",
        repo_type=repo_type,
        commit_message="FoodQA HF upload smoke test summary",
    )
    if not ok:
        raise SystemExit(message)
    print(message)


def _looks_like_placeholder_repo_id(repo_id: str) -> bool:
    value = repo_id.strip().lower()
    placeholders = {"user_or_org/dataset_repo", "dummy-user/dummy-dataset"}
    return value in placeholders or value.startswith("dummy/")


if __name__ == "__main__":
    main()
