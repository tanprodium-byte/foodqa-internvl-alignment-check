from pathlib import Path
import pandas as pd

root = Path(".")
raw_dir = root / "data/raw"
image_dir = root / "data/images"

raw_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)

print("=== RAW FILES ===")
for p in sorted(raw_dir.glob("*")):
    print(p, round(p.stat().st_size / 1024 / 1024, 2), "MB")

csv_path = raw_dir / "food_qa_output_test.csv"
if csv_path.exists():
    print("\n=== CSV ===")
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
else:
    print("\nCSV chưa có:", csv_path)

print("\n=== IMAGES ===")
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
imgs = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
print("Num images:", len(imgs))
for p in imgs[:10]:
    print(p)
