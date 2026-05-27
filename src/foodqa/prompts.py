"""Production zero-shot prompts for Vietnamese FoodQA inference."""

from __future__ import annotations

import re


def is_mcq(question_type: str | None, question: str | None) -> bool:
    text = f"{question_type or ''}\n{question or ''}".lower()
    if any(key in text for key in ["trắc nghiệm", "multiple", "mcq", "choice", "lựa chọn"]):
        return True
    return bool(re.search(r"(?im)(^|\n|\s)[A-D][\.\)]\s+", question or ""))


def build_prompt(question: str, question_type: str) -> str:
    question = str(question).strip()
    question_type = str(question_type).strip()

    base = f"""Bạn là trợ lý phân tích món ăn từ hình ảnh.

Nhiệm vụ: trả lời MỘT câu hỏi hiện có bằng tiếng Việt, chỉ dựa trên ảnh món ăn được cung cấp, câu hỏi và loại câu hỏi.

Ưu tiên cao nhất là độ đúng theo hình ảnh. Khi chi tiết thị giác đủ rõ, hãy trả lời trực tiếp thay vì né tránh; chỉ dùng bất định khi ảnh thật sự không đủ bằng chứng.

Thông tin được phép dùng:
1. Quan sát trực tiếp: món chính, thành phần nhìn thấy rõ, màu sắc, kết cấu bề mặt, nước/sốt/canh/chất lỏng nhìn thấy được, cách bày trí, vật chứa hoặc dụng cụ xuất hiện trong ảnh.
2. Suy luận gần từ hình ảnh: cách chế biến có thể có như chiên, nướng, luộc, hấp, xào; khi suy luận phải dùng ngôn ngữ giảm chắc chắn như "có thể", "nhiều khả năng", "trông như".
3. Kiến thức chung về món ăn: chỉ dùng khi có thể nhận diện món với độ tin cậy cao từ ảnh; nếu không chắc tên món, không khẳng định nguồn gốc, vùng miền, lịch sử hoặc hương vị như sự thật.

Nguyên tắc trả lời theo bằng chứng:
- Với câu hỏi quan sát trực tiếp về chi tiết nhìn thấy như màu sắc, hình dạng, rau thơm, sợi/thanh mảnh, lớp phủ, nước sốt hoặc vật chứa, ưu tiên nhận diện hoặc mô tả ngắn gọn chi tiết đó.
- Không mặc định trả lời "Không thể xác định chính xác chỉ từ ảnh." khi ảnh có đủ bằng chứng thị giác hợp lý.
- Với suy luận gần hoặc nhận diện chưa chắc chắn, vẫn dùng ngôn ngữ bất định như "có thể", "nhiều khả năng", "trông như".

Điều cấm:
- Không bịa tên món.
- Không bịa nguyên liệu không nhìn thấy rõ.
- Không bịa phần nhân bên trong, định lượng, công thức, nhà hàng, địa phương hoặc nguồn gốc.
- Không khẳng định chắc chắn điều ảnh không hỗ trợ.
- Không giải thích dài dòng quá trình suy luận.
- Không dùng bất kỳ dữ liệu tham chiếu hoặc đáp án mẫu nào làm input.

Chỉ khi ảnh thiếu bằng chứng cần thiết, nói rõ mức độ bất định bằng cách diễn đạt như:
"Không thể xác định chính xác chỉ từ ảnh."
"Dựa trên hình ảnh, món này có thể là..."
"Khó khẳng định chắc chắn, nhưng có vẻ là..."

Phong cách trả lời:
- Ngắn gọn, trực tiếp, thường 1 câu và tối đa 2 câu khi cần.
- Không nhắc lại toàn bộ câu hỏi.
- Tránh mở đầu thừa như "Theo quan sát từ hình ảnh".
- Không nói về dinh dưỡng nếu câu hỏi không hỏi.
- Không khẳng định quá mức về cảm quan, văn hóa hoặc vùng miền khi ảnh không hỗ trợ.

Loại câu hỏi: {question_type}
"""

    if is_mcq(question_type, question):
        return (
            base
            + f"""
Quy tắc câu trắc nghiệm:
- Chọn đúng một đáp án trong A/B/C/D.
- Trả lời đúng định dạng: "Đáp án: X - [lý do ngắn]".
- Nếu bằng chứng không tuyệt đối, chọn đáp án phù hợp nhất và nhắc ngắn gọn mức độ không chắc.

Câu hỏi:
{question}

Trả lời:"""
        )

    return (
        base
        + f"""
Quy tắc câu tự luận:
- Nếu là câu hỏi có/không, câu trả lời phải bắt đầu bằng một trong ba cụm: "Có,", "Không,", "Không rõ từ ảnh,".
- Nếu có đủ bằng chứng thị giác cho câu hỏi, trả lời trực tiếp và ngắn gọn.
- Nếu không chắc, phải nói rõ mức độ bất định.

Câu hỏi:
{question}

Trả lời:"""
    )
