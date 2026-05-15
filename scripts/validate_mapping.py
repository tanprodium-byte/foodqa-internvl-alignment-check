from pathlib import Path
import pandas as pd

csv_path = Path("data/raw/food_qa_output_test.csv")
image_dir = Path("data/images")

df = pd.read_csv(csv_path)

available_ids = {
    int(p.stem)
    for p in image_dir.glob("*.jpg")
    if p.stem.isdigit()
}

csv_ids = set(df["image_id"].astype(int).unique())

missing = sorted(csv_ids - available_ids)
extra = sorted(available_ids - csv_ids)

print("CSV rows:", len(df))
print("Unique image_id in CSV:", len(csv_ids))
print("Available images:", len(available_ids))
print("Missing image files:", len(missing))
print("Extra image files:", len(extra))

if missing:
    print("First missing:", missing[:20])

if extra:
    print("First extra:", extra[:20])

print("\nSample rows:")
print(df.head())
