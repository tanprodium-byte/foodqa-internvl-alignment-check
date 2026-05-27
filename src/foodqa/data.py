"""Dataset loading and image-availability checks for the new FoodQA format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "id",
    "image_id",
    "visual_evidence",
    "rationale",
    "question_type",
    "question",
    "answer",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_image_id(x: Any) -> str:
    """Normalize a CSV image identifier to the corresponding image-file stem."""
    if x is None or pd.isna(x):
        return ""

    text = str(x).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return Path(text).stem.strip()


def load_foodqa_csv(path: str | Path) -> pd.DataFrame:
    """Load a FoodQA CSV while preserving IDs as strings."""
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]
    validate_schema(df)
    return df.loc[:, REQUIRED_COLUMNS].copy()


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the new FoodQA schema is present and has no extra columns."""
    columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in columns]
    unexpected = [col for col in columns if col not in REQUIRED_COLUMNS]

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise ValueError("Invalid FoodQA schema; " + "; ".join(parts))


def list_image_files(image_root: str | Path) -> list[Path]:
    """List supported image files recursively under the image root."""
    image_root = Path(image_root)
    if not image_root.exists():
        return []

    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def available_image_ids(image_root: str | Path) -> set[str]:
    """Return normalized image stems available under the image root."""
    return {normalize_image_id(path.stem) for path in list_image_files(image_root)}


def filter_by_available_images(df: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    """Return rows whose image_id has a matching image file under image_root."""
    validate_schema(df)
    available_ids = available_image_ids(image_root)
    normalized_ids = df["image_id"].map(normalize_image_id)
    matched = df.loc[normalized_ids.isin(available_ids), REQUIRED_COLUMNS].copy()
    return matched


def prepare_matched_dataset(
    data_csv: str | Path,
    image_root: str | Path,
    matched_csv: str | Path,
) -> dict[str, Any]:
    """Create the matched CSV and print a compact dataset summary."""
    data_csv = Path(data_csv)
    image_root = Path(image_root)
    matched_csv = Path(matched_csv)

    df = load_foodqa_csv(data_csv)
    image_files = list_image_files(image_root)
    available_ids = {normalize_image_id(path.stem) for path in image_files}
    csv_ids = df["image_id"].map(normalize_image_id)
    unique_csv_ids = {image_id for image_id in csv_ids if image_id}

    matched_mask = csv_ids.isin(available_ids)
    matched_df = df.loc[matched_mask, REQUIRED_COLUMNS].copy()
    missing_ids = sorted(unique_csv_ids - available_ids, key=_natural_sort_key)

    matched_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(matched_csv, index=False, encoding="utf-8-sig")

    question_type_counts = {
        str(key): int(value)
        for key, value in df["question_type"].fillna("").astype(str).str.strip().value_counts().items()
    }

    summary: dict[str, Any] = {
        "total_rows": int(len(df)),
        "unique_csv_image_ids": int(len(unique_csv_ids)),
        "image_file_count": int(len(image_files)),
        "unique_available_image_ids": int(len(available_ids)),
        "matched_rows": int(len(matched_df)),
        "missing_image_id_count": int(len(missing_ids)),
        "first_missing_ids": missing_ids[:20],
        "question_type_counts": question_type_counts,
        "matched_csv": str(matched_csv),
    }

    print_summary(summary)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable preparation summary."""
    print("=== FoodQA New Dataset Summary ===")
    print(f"total rows: {summary['total_rows']}")
    print(f"unique CSV image ids: {summary['unique_csv_image_ids']}")
    print(f"image file count: {summary['image_file_count']}")
    print(f"unique available image ids: {summary['unique_available_image_ids']}")
    print(f"matched rows: {summary['matched_rows']}")
    print(f"missing image id count: {summary['missing_image_id_count']}")
    print(f"first missing ids: {summary['first_missing_ids']}")
    print("question_type counts:")
    for question_type, count in summary["question_type_counts"].items():
        print(f"  {question_type}: {count}")
    print(f"matched CSV: {summary['matched_csv']}")


def _natural_sort_key(text: str) -> tuple[int, int | str]:
    return (0, int(text)) if text.isdigit() else (1, text)
