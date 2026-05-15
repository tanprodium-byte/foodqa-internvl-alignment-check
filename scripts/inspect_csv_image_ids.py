from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/raw/food_qa_output_test.csv")
IMAGE_DIR = Path("data/images")

df = pd.read_csv(CSV_PATH)

print("CSV shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n=== First 30 unique image_id with first QA ===")
sample = df.groupby("image_id").first().reset_index().head(30)

for _, row in sample.iterrows():
    image_id = int(row["image_id"])
    img_path = IMAGE_DIR / f"{image_id}.jpg"
    exists = img_path.exists()

    print("-" * 80)
    print("image_id:", image_id)
    print("image_path:", img_path)
    print("image_exists:", exists)
    print("question:", row["question"])
    print("answer:", row["answer"])

print("\n=== Search suspicious keywords in CSV answers ===")
keywords = ["burger", "hamburger", "phở", "pho", "bún", "gà", "khoai", "tôm", "bánh bao"]

for kw in keywords:
    mask = df["answer"].astype(str).str.lower().str.contains(kw.lower(), na=False)
    hits = df[mask].head(5)
    print("\nKeyword:", kw, "| hits:", int(mask.sum()))
    if len(hits) > 0:
        print(hits[["id", "image_id", "question", "answer"]])