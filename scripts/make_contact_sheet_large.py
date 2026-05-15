from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CSV_PATH = Path("data/raw/food_qa_output_test.csv")
IMAGE_DIR = Path("data/images")
OUT_PATH = Path("outputs/contact_sheet_large_first_12.jpg")


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_wrapped_text(draw, text, xy, font, fill, width_chars, line_spacing=6, max_lines=8):
    x, y = xy
    lines = []
    for raw_line in str(text).splitlines():
        lines.extend(textwrap.wrap(raw_line, width=width_chars))

    for i, line in enumerate(lines[:max_lines]):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_spacing

    if len(lines) > max_lines:
        draw.text((x, y), "...", font=font, fill=fill)

    return y


def main():
    df = pd.read_csv(CSV_PATH)

    # Lấy dòng QA đầu tiên cho mỗi image_id
    sample_rows = df.groupby("image_id").first().reset_index().head(12)

    cols = 3
    cell_w = 520
    img_h = 360
    text_h = 260
    cell_h = img_h + text_h

    rows = (len(sample_rows) + cols - 1) // cols

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(28)
    font_text = load_font(22)
    font_small = load_font(18)

    for idx, row in sample_rows.iterrows():
        image_id = int(row["image_id"])
        image_path = IMAGE_DIR / f"{image_id}.jpg"

        col = idx % cols
        row_idx = idx // cols
        x = col * cell_w
        y = row_idx * cell_h

        # Border
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(180, 180, 180), width=2)

        # Image
        if image_path.exists():
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((cell_w - 20, img_h - 20))
            img_x = x + (cell_w - img.width) // 2
            img_y = y + 10
            canvas.paste(img, (img_x, img_y))
        else:
            draw.rectangle([x + 10, y + 10, x + cell_w - 10, y + img_h - 10], outline="red", width=4)
            draw.text((x + 20, y + 20), f"Missing {image_id}.jpg", font=font_title, fill="red")

        # Text
        text_x = x + 16
        text_y = y + img_h + 12

        draw.text((text_x, text_y), f"id={image_id}", font=font_title, fill="black")
        text_y += 38

        question = str(row["question"])
        answer = str(row["answer"])

        draw.text((text_x, text_y), "Q:", font=font_text, fill=(0, 80, 180))
        text_y = draw_wrapped_text(
            draw,
            question,
            (text_x + 34, text_y),
            font_small,
            "black",
            width_chars=44,
            line_spacing=4,
            max_lines=3,
        )

        text_y += 10
        draw.text((text_x, text_y), "A:", font=font_text, fill=(180, 80, 0))
        draw_wrapped_text(
            draw,
            answer,
            (text_x + 34, text_y),
            font_small,
            "black",
            width_chars=44,
            line_spacing=4,
            max_lines=5,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, quality=95)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()