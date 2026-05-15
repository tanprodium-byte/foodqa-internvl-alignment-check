import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "OpenGVLab/InternVL3_5-2B-Instruct"
IMAGE_GLOB = "data/images/**/*"

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

def find_first_image():
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [
        p for p in glob.glob(IMAGE_GLOB, recursive=True)
        if os.path.splitext(p.lower())[1] in exts
    ]
    if not files:
        raise FileNotFoundError("Không tìm thấy ảnh trong data/images/")
    return files[0]

def main():
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    image_path = find_first_image()
    print("Image:", image_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()

    pixel_values = load_image(image_path).to(torch.float16).cuda()

    question = (
        "<image>\n"
        "Bạn là trợ lý phân tích ảnh món ăn. "
        "Hãy trả lời bằng tiếng Việt. "
        "Chỉ nói những gì nhìn thấy rõ trong ảnh; nếu không chắc, hãy nói 'không rõ từ ảnh'. "
        "Câu hỏi: Món ăn này có những thành phần nào nhìn thấy rõ?"
    )

    generation_config = {
        "max_new_tokens": 256,
        "do_sample": False,
    }

    print("Running inference...")
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    print("\n=== RESPONSE ===")
    print(response)

if __name__ == "__main__":
    main()
