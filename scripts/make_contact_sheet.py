from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw


CSV_PATH = Path("data/raw/food_qa_output_test.csv")
IMAGE_DIR = Path("data/images")
OUT_PATH = Path("outputs/contact_sheet_first_20.jpg")


def main():
    df = pd.read_csv(CSV_PATH)

    # Lấy dòng QA đầu tiên cho mỗi image_id
    sample_rows = df.groupby("image_id").first().reset_index().head(20)

    thumb_w, thumb_h = 260, 190
    text_h = 120
    cols = 4
    rows = 5

    canvas_w = cols * thumb_w
    canvas_h = rows * (thumb_h + text_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, row in sample_rows.iterrows():
        image_id = int(row["image_id"])
        image_path = IMAGE_DIR / f"{image_id}.jpg"

        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + text_h)

        if image_path.exists():
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            canvas.paste(img, (x, y))
        else:
            draw.rectangle(
                [x, y, x + thumb_w - 1, y + thumb_h - 1],
                outline="red",
                width=3,
            )
            draw.text((x + 5, y + 5), f"Missing {image_id}.jpg", fill="red")

        question = str(row["question"])
        answer = str(row["answer"])

        text = f"id={image_id}\nQ: {question}\nA: {answer}"
        wrapped_lines = []
        for line in text.splitlines():
            wrapped_lines.extend(textwrap.wrap(line, width=38))

        text_x = x + 5
        text_y = y + thumb_h + 5

        for j, line in enumerate(wrapped_lines[:7]):
            draw.text((text_x, text_y + j * 15), line, fill="black")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)

    print(f"Saved contact sheet to: {OUT_PATH}")


if __name__ == "__main__":
    main()