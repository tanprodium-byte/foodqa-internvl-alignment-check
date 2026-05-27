"""Small Hugging Face Hub helpers for prediction-file uploads."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def is_hf_available() -> bool:
    """Return whether huggingface_hub can be imported."""
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        return False
    return True


def check_hf_auth(repo_id: str | None = None, repo_type: str = "dataset") -> tuple[bool, str]:
    """Check for a usable Hugging Face token and optional repo access."""
    try:
        from huggingface_hub import HfApi
        try:
            from huggingface_hub import get_token
        except ImportError:
            from huggingface_hub.utils import get_token
    except Exception as exc:
        return False, f"huggingface_hub is not available: {exc!r}"

    api = HfApi()
    try:
        token = get_token()
    except Exception:
        token = None

    if not token:
        return False, "No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN."

    try:
        whoami = api.whoami(token=token)
    except Exception as exc:
        return False, f"Hugging Face authentication failed: {exc!r}"

    username = whoami.get("name") or whoami.get("fullname") or "authenticated user"
    if repo_id:
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        except Exception as exc:
            return False, f"Authenticated as {username}, but repo access failed for {repo_id}: {exc!r}"
        return True, f"Authenticated as {username}; repo access OK for {repo_id}."

    return True, f"Authenticated as {username}."


def upload_prediction_file_to_hf(
    local_path: str | Path,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    commit_message: str | None = None,
) -> tuple[bool, str]:
    """Upload a local prediction artifact to Hugging Face Hub."""
    if not str(path_in_repo).startswith("predictions/"):
        return False, f"Refusing to upload outside predictions/: {path_in_repo}"

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return False, f"huggingface_hub is not available: {exc!r}"

    local_path = Path(local_path)
    if not local_path.exists():
        return False, f"Local file does not exist: {local_path}"

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {path_in_repo}",
        )
    except Exception as exc:
        return False, f"Failed to upload {local_path} to {repo_id}/{path_in_repo}: {exc!r}"

    return True, f"Uploaded {local_path} to {repo_id}/{path_in_repo}."


def upload_json_to_hf(
    payload: dict[str, Any],
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    commit_message: str | None = None,
) -> tuple[bool, str]:
    """Upload a JSON payload through a temporary local file."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False, indent=2)
            temp.write("\n")
            temp_path = Path(temp.name)

        return upload_prediction_file_to_hf(
            local_path=temp_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def prune_remote_prediction_files(
    repo_id: str,
    run_name: str,
    repo_type: str = "dataset",
    keep_checkpoints: bool = False,
) -> tuple[bool, str]:
    """Prune old visible prediction files for a run without rewriting HF history."""
    if not run_name:
        return False, "Cannot prune remote prediction files without run_name."

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return False, f"huggingface_hub is not available: {exc!r}"

    api = HfApi()
    keep_paths = {
        f"predictions/{run_name}.latest.jsonl",
        f"predictions/{run_name}.summary.json",
        f"predictions/{run_name}.final.jsonl",
    }
    deleted: list[str] = []
    warnings: list[str] = []
    folder_pruned = False

    if not keep_checkpoints and hasattr(api, "delete_folder"):
        try:
            api.delete_folder(
                path_in_repo="predictions/checkpoints",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message="Prune prediction checkpoints",
            )
            folder_pruned = True
            deleted.append("predictions/checkpoints/")
        except Exception as exc:
            warnings.append(f"delete_folder failed for predictions/checkpoints/: {exc!r}")

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as exc:
        if folder_pruned:
            return False, "Pruned predictions/checkpoints/, but could not list repo files: " + repr(exc)
        return False, f"Could not list repo files for pruning: {exc!r}"

    candidates: list[str] = []
    for path in repo_files:
        if path in keep_paths:
            continue
        if not keep_checkpoints and path.startswith("predictions/checkpoints/") and not folder_pruned:
            candidates.append(path)
            continue
        if path.startswith(f"predictions/{run_name}."):
            candidates.append(path)

    if not candidates:
        message = "No remote prediction files needed pruning."
        if deleted:
            message = f"Pruned {len(deleted)} remote path(s): {deleted}"
        if warnings:
            return False, message + " Warnings: " + " | ".join(warnings)
        return True, message

    if not hasattr(api, "delete_file"):
        return False, "Hugging Face Hub API does not expose delete_file; pruning skipped."

    for path in candidates:
        try:
            api.delete_file(
                path_in_repo=path,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Prune old prediction file {path}",
            )
            deleted.append(path)
        except Exception as exc:
            warnings.append(f"delete_file failed for {path}: {exc!r}")

    if warnings:
        return False, f"Pruned {len(deleted)} remote path(s), with warnings: " + " | ".join(warnings)
    return True, f"Pruned {len(deleted)} remote path(s): {deleted}"
