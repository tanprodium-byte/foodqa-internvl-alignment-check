# FoodQA InternVL Alignment Check

Zero-shot evaluation of `OpenGVLab/InternVL3_5-2B-Instruct` on a new-format Vietnamese FoodQA dataset.

## Goal

This repository provides a clean, reproducible research workflow for evaluating InternVL3.5-2B on Vietnamese food-image question answering. The model receives one image, one existing question, and the question type, then returns one concise Vietnamese answer.

## Dataset

The expected CSV schema is exactly:

```text
id, image_id, visual_evidence, rationale, question_type, question, answer
```

Place files at:

```text
data/raw/food_qa_output.csv
data/images/
```

Images are discovered recursively under `data/images` by matching `image_id` to image-file stems. Supported extensions are `jpg`, `jpeg`, `png`, and `webp`.

## No-Leak Policy

For zero-shot inference, the prompt uses only:

```text
image + question + question_type
```

The fields `visual_evidence`, `rationale`, and `answer` are reference or ground-truth fields. They are never included in the model prompt and are written only to prediction JSONL records for later analysis.

## Setup on Vast.ai

Install PyTorch separately for CUDA 12.8:

```bash
python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Prepare Matched CSV

```bash
python scripts/prepare_foodqa_new.py --config configs/zero_shot_foodqa_new.yaml
```

This writes:

```text
data/raw/food_qa_output_matched.csv
```

and prints row, image, missing-ID, and question-type counts.

## Run a Small Test

```bash
python scripts/run_zero_shot_internvl.py --config configs/zero_shot_foodqa_new.yaml --limit 50
```

The default output path is:

```text
outputs/zero_shot_foodqa_new_test50.jsonl
```

On an A4000 16GB GPU, `max_dynamic_patch=8` has completed a 50-record smoke run with 0 errors. If a per-sample CUDA out-of-memory error occurs, the runner clears CUDA cache and retries that sample with lower patch counts in this order: requested value, then `6`, `4`, `2`, and `1` where those values are lower than requested. If inference still runs out of memory, reduce `max_dynamic_patch` and `max_new_tokens`.

## Full-run logging, resume, and Hugging Face latest-only upload

Long FoodQA zero-shot runs write JSONL locally as the source of truth. Each record is written and flushed immediately, includes `row_index`, and records the requested patch count, the successful patch count, fallback status, elapsed seconds, UTC timestamp, and model name.

Resume is enabled with `--resume`. The runner reads the existing JSONL, treats records with a non-null `pred_answer` or non-null `error` as complete, and skips by `id` first with `row_index` as the fallback key. This matters because one image can have multiple questions.

Hugging Face upload is optional and disabled by default. When enabled, it uploads prediction JSONL artifacts only. It does not upload images, the raw CSV dataset, or `data/`. By default, no checkpoint tree is kept on Hugging Face, and the visible repo file tree uses only:

```text
predictions/<run_name>.latest.jsonl
predictions/<run_name>.summary.json
predictions/<run_name>.final.jsonl
```

`--hf-keep-checkpoints` is optional and off by default. `--hf-prune-remote` can clean old remote checkpoint files and older files for the same run name that are not the latest, summary, or final artifacts. Hugging Face git history is not rewritten; pruning only keeps the visible file tree clean with latest/final outputs. Upload requires `huggingface-cli login` or `HF_TOKEN` in the environment.

Full local run with resume:

```bash
python scripts/run_zero_shot_internvl.py \
  --config configs/zero_shot_foodqa_new.yaml \
  --max-dynamic-patch 8 \
  --output outputs/zero_shot_full_patch8.jsonl \
  --resume \
  --run-name zero_shot_full_patch8
```

Full run with latest-only HF upload:

```bash
python scripts/run_zero_shot_internvl.py \
  --config configs/zero_shot_foodqa_new.yaml \
  --max-dynamic-patch 8 \
  --output outputs/zero_shot_full_patch8.jsonl \
  --resume \
  --run-name zero_shot_full_patch8 \
  --hf-upload \
  --hf-repo-id USER_OR_ORG/DATASET_REPO \
  --hf-upload-every-records 1000 \
  --hf-upload-every-minutes 30 \
  --hf-prune-remote
```

HF dry run:

```bash
python scripts/run_zero_shot_internvl.py \
  --config configs/zero_shot_foodqa_new.yaml \
  --hf-dry-run \
  --hf-repo-id USER_OR_ORG/DATASET_REPO \
  --output outputs/zero_shot_full_patch8.jsonl \
  --run-name zero_shot_full_patch8
```

HF test upload:

```bash
python scripts/run_zero_shot_internvl.py \
  --config configs/zero_shot_foodqa_new.yaml \
  --hf-test-upload \
  --hf-repo-id USER_OR_ORG/DATASET_REPO
```

## View Predictions

```bash
python scripts/view_predictions.py --pred outputs/zero_shot_foodqa_new_test50.jsonl --limit 20
```

The viewer prints a deterministic metrics summary and readable examples with ID, image ID, question type, question, ground truth, prediction, and per-record error.

## Repository Notes

Do not commit data, model caches, or full inference outputs. Keep large raw data under `data/`, Hugging Face caches under ignored cache directories, and runtime JSONL outputs under `outputs/`.

The production prompt is implemented in `src/foodqa/prompts.py` and follows `docs/foodqa_prompt_reference.md`. It is a zero-shot answering prompt, not the original QA-generation prompt.
