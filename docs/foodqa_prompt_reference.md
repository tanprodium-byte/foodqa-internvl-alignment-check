# FoodQA Prompt Reference

This file documents the original FoodQA prompt principles used to generate the new Vietnamese FoodQA dataset.

The original prompt is a QA-generation prompt, not an inference prompt. For zero-shot evaluation, we must not ask the model to generate 25 QA pairs. Instead, the model receives one image and one existing question, then returns one answer.

## Original FoodQA principles to preserve

### Highest priority

The highest priority is correctness grounded in the food image. Being honest about uncertainty is more important than guessing.

### Allowed information

1. Direct visual observation:
   - main dish
   - clearly visible ingredients
   - colors
   - surface texture
   - visible sauce, soup, broth, or liquid
   - presentation
   - visible container or utensils

2. Near visual inference:
   - possible cooking method such as fried, grilled, boiled, steamed, or stir-fried
   - must use uncertainty language such as:
     - "có thể"
     - "nhiều khả năng"
     - "trông như"

3. General food knowledge:
   - only when the dish can be identified with high confidence from the image
   - if dish identity is uncertain, do not assert origin, region, history, or flavor as fact

### Forbidden behavior

- Do not invent dish names.
- Do not invent ingredients that are not clearly visible.
- Do not invent hidden fillings, quantities, recipes, restaurants, localities, or origins.
- Do not confidently assert anything not supported by the image.
- Do not provide long reasoning.

### Uncertainty handling

If the image is not sufficient, clearly express uncertainty.

Safe Vietnamese uncertainty expressions:
- "Không thể xác định chính xác chỉ từ ảnh."
- "Dựa trên hình ảnh, món này có thể là..."
- "Khó khẳng định chắc chắn, nhưng có vẻ là..."

### Yes/no questions

For yes/no questions, the answer must begin with exactly one of:
- "Có,"
- "Không,"
- "Không rõ từ ảnh,"

### Answer style

- Answer concisely and directly.
- Prefer one sentence.
- Use two sentences only when necessary.
- Do not repeat the full question.
- Avoid filler phrases such as "Theo quan sát từ hình ảnh".
- Prioritize core information.
- Do not mention nutrition unless explicitly asked.
- Do not over-claim sensory, cultural, or regional facts when the image does not support them.

### MCQ questions

For multiple-choice questions:
- Choose exactly one answer from A/B/C/D.
- The wrong options may be plausible.
- The model output must follow:
  "Đáp án: X - [lý do ngắn]"

## Zero-shot no-leak policy

The new FoodQA dataset has this schema:

id, image_id, visual_evidence, rationale, question_type, question, answer

For zero-shot inference, the model input must use only:

image + question + question_type

Do not include these fields in the prompt:
- visual_evidence
- rationale
- answer

These fields are reference/ground-truth data and should be kept only in the output JSONL for later analysis.

## Required zero-shot prompt behavior

The production prompt in src/foodqa/prompts.py should be a zero-shot answering prompt, not the original QA-generation prompt.

It should ask the model to answer ONE existing Vietnamese question from ONE food image.

It should preserve the safety and grounding principles above while avoiding the original dataset-generation instructions such as:
- generating at least 25 QA pairs
- returning a JSON array
- creating new questions