#!/usr/bin/env python3
"""Prepare the matched new-format FoodQA CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from foodqa.data import prepare_matched_dataset  # noqa: E402
from foodqa.internvl_runner import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_matched_dataset(
        data_csv=config["data_csv"],
        image_root=config["image_root"],
        matched_csv=config["matched_csv"],
    )


if __name__ == "__main__":
    main()
