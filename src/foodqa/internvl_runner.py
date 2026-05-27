"""InternVL zero-shot runner for the new Vietnamese FoodQA format."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .data import load_foodqa_csv
from .prompts import build_prompt


DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-2B-Instruct"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_torch(disable_cudnn: bool = True):
    import torch

    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    return torch


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "float16",
    use_flash_attn: bool = False,
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = _resolve_torch_dtype(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    return model, tokenizer


def run_zero_shot(
    data_csv: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    limit: int | None = None,
    input_size: int = 448,
    max_dynamic_patch: int = 2,
    max_new_tokens: int = 128,
    torch_dtype: str = "float16",
    disable_cudnn: bool = True,
    use_flash_attn: bool = False,
    batch_size: int = 1,
    resume: bool = False,
    sample_distinct_images: int | None = None,
    seed: int = 42,
    hf_upload: bool = False,
    hf_repo_id: str | None = None,
    hf_repo_type: str = "dataset",
    hf_upload_every_records: int = 1000,
    hf_upload_every_minutes: float = 30.0,
    hf_keep_checkpoints: bool = False,
    hf_prune_remote: bool = False,
    run_name: str | None = None,
) -> dict[str, Any]:
    if batch_size != 1:
        raise ValueError("InternVL FoodQA runner supports batch_size=1 only.")

    data_csv = Path(data_csv)
    image_root = Path(image_root)
    output_path = Path(output_path)
    run_name = run_name or output_path.stem
    requested_patch = int(max_dynamic_patch)

    if hf_upload:
        if not hf_repo_id:
            raise ValueError("hf_upload=True requires hf_repo_id.")
        from .hf_utils import check_hf_auth

        ok, message = check_hf_auth(repo_id=hf_repo_id, repo_type=hf_repo_type)
        if not ok:
            raise RuntimeError(f"Hugging Face upload requested, but auth/access check failed: {message}")
        print(f"HF auth: {message}")

    from .image_utils import find_image, load_image

    df = load_foodqa_csv(data_csv)
    if sample_distinct_images is not None and sample_distinct_images > 0:
        df = _sample_distinct_images(df, sample_distinct_images, seed)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys: set[str] = set()
    malformed_existing_lines = 0
    if resume and output_path.exists():
        completed_keys, malformed_existing_lines = _load_completed_record_keys(output_path)
        _ensure_jsonl_trailing_newline(output_path)

    rows_to_process = []
    for row_index, row in df.iterrows():
        key = _record_key_from_row(row, row_index)
        if resume and key in completed_keys:
            continue
        rows_to_process.append((row_index, row))

    if resume:
        print("=== Resume State ===")
        print(f"completed count: {len(df) - len(rows_to_process)}")
        print(f"remaining count: {len(rows_to_process)}")
        print(f"output path: {output_path}")
        if malformed_existing_lines:
            print(f"warning: ignored {malformed_existing_lines} malformed existing JSONL line(s).")

    print("=== InternVL Zero-Shot FoodQA ===")
    print(f"data CSV: {data_csv}")
    print(f"image root: {image_root}")
    print(f"output path: {output_path}")
    print(f"run_name: {run_name}")
    print(f"model: {model_name}")
    print(f"rows requested: {len(df)}")
    print(f"rows remaining: {len(rows_to_process)}")
    print(f"input_size: {input_size}")
    print(f"max_dynamic_patch: {requested_patch}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"torch_dtype: {torch_dtype}")
    if sample_distinct_images is not None and sample_distinct_images > 0:
        print(f"sample_distinct_images: {sample_distinct_images}")
        print(f"seed: {seed}")
    if hf_upload:
        print(f"HF upload: enabled -> {hf_repo_id}")
        print(f"HF latest path: predictions/{run_name}.latest.jsonl")
        print(f"HF summary path: predictions/{run_name}.summary.json")
        print(f"HF final path: predictions/{run_name}.final.jsonl")
        print(f"HF keep checkpoints: {hf_keep_checkpoints}")
        print(f"HF prune remote: {hf_prune_remote}")

    written = 0
    errors = 0
    latest_upload_record_count = len(completed_keys)
    last_upload_written = 0
    last_upload_time = time.monotonic()

    if not rows_to_process:
        summary = _build_run_summary(
            run_name=run_name,
            model_name=model_name,
            output_path=output_path,
            records_written_this_run=written,
            records_with_error_this_run=errors,
            total_completed_records_in_output=len(completed_keys),
            requested_max_dynamic_patch=requested_patch,
            finished=True,
            latest_upload_record_count=latest_upload_record_count,
        )
        if hf_upload:
            _upload_final_artifacts(
                output_path=output_path,
                summary=summary,
                repo_id=hf_repo_id,
                repo_type=hf_repo_type,
                run_name=run_name,
                hf_prune_remote=hf_prune_remote,
                hf_keep_checkpoints=hf_keep_checkpoints,
            )
        _print_summary(summary)
        return summary

    torch = setup_torch(disable_cudnn=disable_cudnn)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL inference, but torch.cuda.is_available() is false.")
    dtype = _resolve_torch_dtype(torch, torch_dtype)

    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        torch_dtype=torch_dtype,
        use_flash_attn=use_flash_attn,
    )

    generation_config = {
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens),
    }
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_config["pad_token_id"] = tokenizer.pad_token_id

    mode = "a" if resume and output_path.exists() else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for row_index, row in tqdm(rows_to_process, total=len(rows_to_process)):
            record_start = time.perf_counter()
            image_id = str(row.get("image_id", "")).strip()
            question_type = str(row.get("question_type", "")).strip()
            question = str(row.get("question", "")).strip()
            image_path = find_image(image_root, image_id)

            record = _build_base_record(
                row=row,
                row_index=row_index,
                image_id=image_id,
                image_path=image_path,
                question_type=question_type,
                question=question,
                requested_patch=requested_patch,
                model_name=model_name,
            )

            if image_path is None:
                record["error"] = "image_not_found"
                record["elapsed_sec"] = round(time.perf_counter() - record_start, 6)
                record["timestamp_utc"] = _utc_now()
                errors += 1
                _write_jsonl(f, record)
                written += 1
                completed_keys.add(_record_key_from_record(record))
                latest_upload_record_count = len(completed_keys)
                if hf_upload and _should_upload(
                    written=written,
                    last_upload_written=last_upload_written,
                    last_upload_time=last_upload_time,
                    every_records=hf_upload_every_records,
                    every_minutes=hf_upload_every_minutes,
                ):
                    latest_upload_record_count = _periodic_hf_upload(
                        output_path=output_path,
                        repo_id=hf_repo_id,
                        repo_type=hf_repo_type,
                        run_name=run_name,
                        model_name=model_name,
                        written=written,
                        errors=errors,
                        total_completed=len(completed_keys),
                        requested_patch=requested_patch,
                        hf_keep_checkpoints=hf_keep_checkpoints,
                    )
                    last_upload_written = written
                    last_upload_time = time.monotonic()
                continue

            prompt = build_prompt(question=question, question_type=question_type)
            pred_answer, used_patch, error = _predict_with_oom_fallback(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                prompt=prompt,
                generation_config=generation_config,
                input_size=input_size,
                requested_patch=requested_patch,
                dtype=dtype,
                load_image=load_image,
            )

            record["pred_answer"] = pred_answer
            record["used_max_dynamic_patch"] = used_patch
            record["fallback_used"] = used_patch is not None and used_patch != requested_patch
            record["error"] = error
            record["elapsed_sec"] = round(time.perf_counter() - record_start, 6)
            record["timestamp_utc"] = _utc_now()
            if error is not None:
                errors += 1

            _write_jsonl(f, record)
            written += 1
            completed_keys.add(_record_key_from_record(record))
            latest_upload_record_count = len(completed_keys)

            if hf_upload and _should_upload(
                written=written,
                last_upload_written=last_upload_written,
                last_upload_time=last_upload_time,
                every_records=hf_upload_every_records,
                every_minutes=hf_upload_every_minutes,
            ):
                latest_upload_record_count = _periodic_hf_upload(
                    output_path=output_path,
                    repo_id=hf_repo_id,
                    repo_type=hf_repo_type,
                    run_name=run_name,
                    model_name=model_name,
                    written=written,
                    errors=errors,
                    total_completed=len(completed_keys),
                    requested_patch=requested_patch,
                    hf_keep_checkpoints=hf_keep_checkpoints,
                )
                last_upload_written = written
                last_upload_time = time.monotonic()

    summary = _build_run_summary(
        run_name=run_name,
        model_name=model_name,
        output_path=output_path,
        records_written_this_run=written,
        records_with_error_this_run=errors,
        total_completed_records_in_output=len(completed_keys),
        requested_max_dynamic_patch=requested_patch,
        finished=True,
        latest_upload_record_count=latest_upload_record_count,
    )

    if hf_upload:
        _upload_final_artifacts(
            output_path=output_path,
            summary=summary,
            repo_id=hf_repo_id,
            repo_type=hf_repo_type,
            run_name=run_name,
            hf_prune_remote=hf_prune_remote,
            hf_keep_checkpoints=hf_keep_checkpoints,
        )

    _print_summary(summary)
    return summary


def _sample_distinct_images(df, sample_distinct_images: int, seed: int):
    import random

    groups: dict[str, list[Any]] = {}
    for row_index, row in df.iterrows():
        image_id = str(row.get("image_id", "")).strip()
        groups.setdefault(image_id, []).append(row_index)

    rng = random.Random(seed)
    image_ids = sorted(groups)
    rng.shuffle(image_ids)
    selected_indices = []
    for image_id in image_ids[:sample_distinct_images]:
        selected_indices.append(rng.choice(groups[image_id]))

    return df.loc[selected_indices].copy()


def _build_base_record(
    *,
    row,
    row_index: Any,
    image_id: str,
    image_path: Path | None,
    question_type: str,
    question: str,
    requested_patch: int,
    model_name: str,
) -> dict[str, Any]:
    return {
        "row_index": int(row_index) if _is_int_like(row_index) else row_index,
        "id": str(row.get("id", "")).strip(),
        "image_id": image_id,
        "image_path": str(image_path) if image_path else None,
        "question_type": question_type,
        "question": question,
        "gt_answer": str(row.get("answer", "")).strip(),
        "visual_evidence": str(row.get("visual_evidence", "")).strip(),
        "rationale": str(row.get("rationale", "")).strip(),
        "pred_answer": None,
        "error": None,
        "requested_max_dynamic_patch": requested_patch,
        "used_max_dynamic_patch": None,
        "fallback_used": False,
        "elapsed_sec": None,
        "timestamp_utc": None,
        "model_name": model_name,
    }


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _predict_with_oom_fallback(
    *,
    torch,
    model,
    tokenizer,
    image_path: Path,
    prompt: str,
    generation_config: dict[str, Any],
    input_size: int,
    requested_patch: int,
    dtype,
    load_image,
) -> tuple[str | None, int | None, str | None]:
    final_error: str | None = None
    for patch in _fallback_patch_order(requested_patch):
        try:
            pred_answer = _predict_one(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                prompt=prompt,
                generation_config=generation_config,
                input_size=input_size,
                max_dynamic_patch=patch,
                dtype=dtype,
                load_image=load_image,
            )
            return pred_answer, patch, None
        except RuntimeError as exc:
            final_error = repr(exc)
            if _is_oom_error(exc):
                torch.cuda.empty_cache()
                continue
            return None, None, final_error
        except Exception as exc:
            return None, None, repr(exc)

    return None, None, final_error or "CUDA OOM retry attempts exhausted"


def _predict_one(
    *,
    torch,
    model,
    tokenizer,
    image_path: Path,
    prompt: str,
    generation_config: dict[str, Any],
    input_size: int,
    max_dynamic_patch: int,
    dtype,
    load_image,
) -> str:
    pixel_values = load_image(
        image_path,
        input_size=input_size,
        max_dynamic_patch=max_dynamic_patch,
    ).to(device="cuda", dtype=dtype)

    with torch.inference_mode():
        return model.chat(
            tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )


def _fallback_patch_order(requested_patch: int) -> list[int]:
    order = [requested_patch]
    for patch in [6, 4, 2, 1]:
        if patch < requested_patch and patch not in order:
            order.append(patch)
    return order


def _should_upload(
    *,
    written: int,
    last_upload_written: int,
    last_upload_time: float,
    every_records: int,
    every_minutes: float,
) -> bool:
    record_due = every_records > 0 and (written - last_upload_written) >= every_records
    minute_due = every_minutes >= 0 and (time.monotonic() - last_upload_time) >= every_minutes * 60.0
    return record_due or minute_due


def _periodic_hf_upload(
    *,
    output_path: Path,
    repo_id: str,
    repo_type: str,
    run_name: str,
    model_name: str,
    written: int,
    errors: int,
    total_completed: int,
    requested_patch: int,
    hf_keep_checkpoints: bool,
) -> int:
    from .hf_utils import upload_json_to_hf, upload_prediction_file_to_hf

    latest_path = f"predictions/{run_name}.latest.jsonl"
    summary_path = f"predictions/{run_name}.summary.json"
    summary = _build_run_summary(
        run_name=run_name,
        model_name=model_name,
        output_path=output_path,
        records_written_this_run=written,
        records_with_error_this_run=errors,
        total_completed_records_in_output=total_completed,
        requested_max_dynamic_patch=requested_patch,
        finished=False,
        latest_upload_record_count=total_completed,
    )

    ok, message = upload_prediction_file_to_hf(
        local_path=output_path,
        repo_id=repo_id,
        path_in_repo=latest_path,
        repo_type=repo_type,
        commit_message=f"Update latest predictions for {run_name}",
    )
    _print_hf_result(ok, message)

    ok, message = upload_json_to_hf(
        payload=summary,
        repo_id=repo_id,
        path_in_repo=summary_path,
        repo_type=repo_type,
        commit_message=f"Update prediction summary for {run_name}",
    )
    _print_hf_result(ok, message)

    if hf_keep_checkpoints:
        checkpoint_path = f"predictions/checkpoints/{run_name}.step_{total_completed}.jsonl"
        ok, message = upload_prediction_file_to_hf(
            local_path=output_path,
            repo_id=repo_id,
            path_in_repo=checkpoint_path,
            repo_type=repo_type,
            commit_message=f"Upload prediction checkpoint for {run_name}",
        )
        _print_hf_result(ok, message)

    return total_completed


def _upload_final_artifacts(
    *,
    output_path: Path,
    summary: dict[str, Any],
    repo_id: str,
    repo_type: str,
    run_name: str,
    hf_prune_remote: bool,
    hf_keep_checkpoints: bool,
) -> None:
    from .hf_utils import prune_remote_prediction_files, upload_json_to_hf, upload_prediction_file_to_hf

    ok, message = upload_prediction_file_to_hf(
        local_path=output_path,
        repo_id=repo_id,
        path_in_repo=f"predictions/{run_name}.final.jsonl",
        repo_type=repo_type,
        commit_message=f"Upload final predictions for {run_name}",
    )
    _print_hf_result(ok, message)

    ok, message = upload_prediction_file_to_hf(
        local_path=output_path,
        repo_id=repo_id,
        path_in_repo=f"predictions/{run_name}.latest.jsonl",
        repo_type=repo_type,
        commit_message=f"Update latest predictions for {run_name}",
    )
    _print_hf_result(ok, message)

    ok, message = upload_json_to_hf(
        payload=summary,
        repo_id=repo_id,
        path_in_repo=f"predictions/{run_name}.summary.json",
        repo_type=repo_type,
        commit_message=f"Upload prediction summary for {run_name}",
    )
    _print_hf_result(ok, message)

    if hf_prune_remote:
        ok, message = prune_remote_prediction_files(
            repo_id=repo_id,
            run_name=run_name,
            repo_type=repo_type,
            keep_checkpoints=hf_keep_checkpoints,
        )
        _print_hf_result(ok, message, operation="HF prune")


def _print_hf_result(ok: bool, message: str, operation: str = "HF upload") -> None:
    if ok:
        print(f"{operation}: {message}")
    else:
        print(f"warning: {operation} failed: {message}")


def _build_run_summary(
    *,
    run_name: str,
    model_name: str,
    output_path: Path,
    records_written_this_run: int,
    records_with_error_this_run: int,
    total_completed_records_in_output: int,
    requested_max_dynamic_patch: int,
    finished: bool,
    latest_upload_record_count: int,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "model_name": model_name,
        "output_path": str(output_path),
        "records_written_this_run": records_written_this_run,
        "records_with_error_this_run": records_with_error_this_run,
        "total_completed_records_in_output": total_completed_records_in_output,
        "requested_max_dynamic_patch": requested_max_dynamic_patch,
        "timestamp_utc": _utc_now(),
        "finished": finished,
        "latest_upload_record_count": latest_upload_record_count,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("=== Run Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def _load_completed_record_keys(output_path: Path) -> tuple[set[str], int]:
    completed: set[str] = set()
    malformed = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("pred_answer") is not None or record.get("error") is not None:
                key = _record_key_from_record(record)
                if key:
                    completed.add(key)
    return completed, malformed


def _record_key_from_row(row, row_index: Any) -> str:
    record_id = _clean_record_id(row.get("id", ""))
    if record_id:
        return f"id:{record_id}"
    return f"row_index:{row_index}"


def _record_key_from_record(record: dict[str, Any]) -> str:
    record_id = _clean_record_id(record.get("id", ""))
    if record_id:
        return f"id:{record_id}"
    return f"row_index:{record.get('row_index')}"


def _ensure_jsonl_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            f.write(b"\n")
            f.flush()


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _clean_record_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _resolve_torch_dtype(torch, torch_dtype: str):
    dtype_name = str(torch_dtype).lower()
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
