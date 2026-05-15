from pathlib import Path
import pandas as pd
import torch

# Workaround cho lỗi:
# RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZE
# Lỗi này xảy ra ở vision encoder khi chạy Conv2D trên một số môi trường CUDA/cuDNN.
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "OpenGVLab/InternVL3_5-2B-Instruct"
CSV_PATH = Path("data/raw/food_qa_output_test.csv")
IMAGE_DIR = Path("data/images")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    return transform(image).unsqueeze(0)

def find_first_valid_row():
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        image_id = int(row["image_id"])
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        if image_path.exists():
            return row, image_path
    raise FileNotFoundError("Không tìm thấy row nào có ảnh tương ứng.")

def main():
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    row, image_path = find_first_valid_row()
    question_text = str(row["question"])
    gt_answer = str(row["answer"])

    print("\nImage:", image_path)
    print("Question:", question_text)
    print("Ground truth:", gt_answer)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()

    pixel_values = load_image(image_path).to(torch.float16).cuda()

    prompt = (
        "<image>\n"
        "Bạn là trợ lý phân tích ảnh món ăn. "
        "Hãy trả lời bằng tiếng Việt. "
        "Chỉ dựa vào những gì nhìn thấy rõ trong ảnh; nếu không chắc, hãy nói 'không rõ từ ảnh'.\n"
        f"Câu hỏi: {question_text}"
    )

    generation_config = {
        "max_new_tokens": 256,
        "do_sample": False,
    }

    print("\nRunning inference...")
    with torch.inference_mode():
        pred = model.chat(tokenizer, pixel_values, prompt, generation_config)

    print("\n=== MODEL ANSWER ===")
    print(pred)

    print("\n=== GROUND TRUTH ===")
    print(gt_answer)

if __name__ == "__main__":
    main()
