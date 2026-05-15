from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CSV_PATH = Path("data/raw/food_qa_output_test.csv")
IMAGE_DIR = Path("data/images")
OUT_DIR = Path("outputs/keyword_sheets")


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_wrapped(draw, text, x, y, font, width=42, max_lines=5):
    lines = []
    for raw in str(text).splitlines():
        lines.extend(textwrap.wrap(raw, width=width))

    for line in lines[:max_lines]:
        draw.text((x, y), line, font=font, fill="black")
        y += font.size + 5

    if len(lines) > max_lines:
        draw.text((x, y), "...", font=font, fill="black")

    return y


def make_sheet(df, keyword, max_items=12):
    mask = df["answer"].astype(str).str.lower().str.contains(keyword.lower(), na=False)
    hits = df[mask].groupby("image_id").first().reset_index().head(max_items)

    print(f"\nKeyword: {keyword}")
    print("Num QA hits:", int(mask.sum()))
    print("Num unique image hits shown:", len(hits))
    print(hits[["id", "image_id", "question", "answer"]].head())

    cols = 3
    cell_w = 520
    img_h = 360
    text_h = 250
    cell_h = img_h + text_h
    rows = max(1, (len(hits) + cols - 1) // cols)

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(28)
    font_text = load_font(18)

    for idx, row in hits.iterrows():
        image_id = int(row["image_id"])
        img_path = IMAGE_DIR / f"{image_id}.jpg"

        col = idx % cols
        row_idx = idx // cols
        x = col * cell_w
        y = row_idx * cell_h

        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180), width=2)

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((cell_w - 20, img_h - 20))
            canvas.paste(img, (x + (cell_w - img.width) // 2, y + 10))
        else:
            draw.text((x + 20, y + 20), f"Missing {image_id}.jpg", font=font_title, fill="red")

        text_x = x + 16
        text_y = y + img_h + 12

        draw.text((text_x, text_y), f"id={image_id}", font=font_title, fill="black")
        text_y += 42

        draw.text((text_x, text_y), "A:", font=font_title, fill=(180, 80, 0))
        draw_wrapped(draw, row["answer"], text_x + 36, text_y + 4, font_text, width=46, max_lines=7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"keyword_{keyword}.jpg"
    canvas.save(out_path, quality=95)
    print("Saved:", out_path)


def main():
    df = pd.read_csv(CSV_PATH)

    for keyword in ["burger", "hamburger", "bánh bao", "tôm", "gà", "bún bò", "bánh cuốn"]:
        make_sheet(df, keyword)


if __name__ == "__main__":
    main()