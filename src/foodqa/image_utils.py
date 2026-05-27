"""InternVL-style image preprocessing and recursive image lookup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Iterable[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    if min_num < 1 or max_num < min_num:
        raise ValueError("Expected 1 <= min_num <= max_num.")

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.BICUBIC)
    processed_images = []
    grid_width = target_width // image_size

    for i in range(blocks):
        box = (
            (i % grid_width) * image_size,
            (i // grid_width) * image_size,
            ((i % grid_width) + 1) * image_size,
            ((i // grid_width) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.BICUBIC))

    return processed_images


def load_image(
    image_file: str | Path,
    input_size: int = 448,
    max_dynamic_patch: int = 2,
) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_dynamic_patch,
        use_thumbnail=True,
    )
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def find_image(image_root: str | Path, image_id: str) -> Path | None:
    """Find an image by matching image_id to a recursive image-file stem."""
    root = Path(image_root)
    stem = _normalize_stem(image_id)
    if not stem:
        return None

    index = _image_index(str(root.resolve()))
    return index.get(stem)


@lru_cache(maxsize=16)
def _image_index(image_root: str) -> dict[str, Path]:
    root = Path(image_root)
    if not root.exists():
        return {}

    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(_normalize_stem(path.stem), path)
    return index


def _normalize_stem(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return Path(text).stem.strip()
