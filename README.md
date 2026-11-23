# Masri Translate

Fine-tune **Qwen2.5-1.5B-Instruct** to translate between **English** and **Egyptian Arabic** using LLaMA-Factory.

---

## 1. Quick summary

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA SFT (`r=64`, `alpha=128`, dropout `0.1`) |
| Dataset | `oddadmix/egyptian-arabic-english-translation-50k` |
| Directions | English → Egyptian Arabic, Egyptian Arabic → English |
| Evaluation size | 3000 held-out examples per direction |
| Reported metrics | sacreBLEU and chrF++ |

The fine-tuned adapter clearly outperforms the base model in both directions. See [Results](#6-results).

---

## 2. What this repo does

1. Downloads and cleans the parallel dataset.
2. Removes duplicates, invalid pairs and overly long examples.
3. Creates train, validation and two test files.
4. Converts training pairs into Alpaca format for LLaMA-Factory.
5. Trains a LoRA adapter.
6. Generates translations for both directions.
7. Scores them with BLEU and chrF++.

---

## 3. Setup

### 3.1 Install dependencies

Python **3.10+** is required. A CUDA GPU is strongly recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,train,metrics]"
```

If you only want to run tests or read the code, this lighter install is enough:

```bash
pip install -e ".[dev]"
```

### 3.2 Create your `.env` file

```bash
cp .env.example .env
```

Then fill it in:

```dotenv
DATA_NAME="oddadmix/egyptian-arabic-english-translation-50k"
BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_MAX_LENGTH=1024

WANDB_API_KEY="your-wandb-key"
HF_TOKEN="your-huggingface-token"
PROJECT_NAME="masri-translation"

VAL_SIZE=0.03
TEST_SIZE=0.1

PREDICTIONS_AR="saves/exp10/eval_en2ar/generated_predictions.jsonl"
PREDICTIONS_EN="saves/exp10/eval_ar2en/generated_predictions.jsonl"
```

`PREDICTIONS_AR` points to English → Egyptian Arabic predictions.
`PREDICTIONS_EN` points to Egyptian Arabic → English predictions.

---

## 4. Run the pipeline

### Step 1 — Prepare the data

```bash
python -m src.prepare
```

This creates:

```text
data/train.json
data/val.json
data/test_en2ar.json
data/test_ar2en.json
```

These files are gitignored because they are generated locally.

### Step 2 — Train the LoRA adapter

```bash
llamafactory-cli train configs/train.yaml
```

The adapter is saved to:

```text
saves/exp10/lora
```

### Step 3 — Generate translations

```bash
llamafactory-cli train configs/eval_en2ar.yaml
llamafactory-cli train configs/eval_ar2en.yaml
```

> Prediction also uses `llamafactory-cli train` because the eval configs contain `do_predict: true`. This is normal for LLaMA-Factory.

Both configs load:

```text
base model: Qwen/Qwen2.5-1.5B-Instruct
adapter:    saves/exp10/lora
```

Decoding settings are identical for both directions:

```yaml
max_samples: 3000
do_sample: false
num_beams: 4
max_new_tokens: 1024
repetition_penalty: 1.05
fp16: true
```

### Step 4 — Compute BLEU and chrF++

```bash
python -m src.evaluate
```

This reads the two prediction files listed in `.env`, prints the scores, and writes `metrics.json` next to each prediction file.

---

## 5. How to compare fine-tuned vs base fairly

Use the same test data and same decoding settings. The only difference should be the adapter.

| | Fine-tuned run | Base run |
|---|---|---|
| Model | base + LoRA adapter | base only |
| Config | `configs/eval_en2ar.yaml` / `eval_ar2en.yaml` | copy of those configs **without** `adapter_name_or_path` |
| Adapter line | `adapter_name_or_path: saves/exp10/lora` | delete this line |
| Output folder | `saves/exp10/eval_*` | e.g. `saves/base/eval_*` |
| Samples | 3000 | 3000 |
| Decoding | beams=4, no sampling | beams=4, no sampling |

Example for the base run:

```bash
cp configs/eval_en2ar.yaml configs/base_eval_en2ar.yaml
cp configs/eval_ar2en.yaml configs/base_eval_ar2en.yaml
```

Then in both new files:

1. delete the `adapter_name_or_path: ...` line
2. change `output_dir` to a new folder, e.g. `saves/base/eval_en2ar`

Before scoring the base run, update `.env`:

```dotenv
PREDICTIONS_AR="saves/base/eval_en2ar/generated_predictions.jsonl"
PREDICTIONS_EN="saves/base/eval_ar2en/generated_predictions.jsonl"
```

Then run:

```bash
python -m src.evaluate
```

### Do I need to merge LoRA weights?

**No, not for evaluation.** Your fine-tuned model is already `base + adapter`, which LLaMA-Factory loads automatically.

Merging is only useful if you want a standalone model folder that can be loaded without LLaMA-Factory. If you merge, evaluate the merged model **alone** and remove `adapter_name_or_path`. Do not apply the adapter again on top of merged weights.

---

## 6. Results

Evaluated on the held-out test sets with `max_samples=3000` per direction.

| Model | English → Egyptian Arabic BLEU | English → Egyptian Arabic chrF++ | Egyptian Arabic → English BLEU | Egyptian Arabic → English chrF++ |
|---|---:|---:|---:|---:|
| Base `Qwen/Qwen2.5-1.5B-Instruct` | 1.91 | 15.21 | 18.41 | 44.26 |
| Fine-tuned LoRA (`saves/exp10/lora`) | **36.39** | **59.75** | **48.72** | **67.20** |

### Extra LLaMA-Factory prediction metrics

These are printed during prediction. They are useful diagnostics, but the table above is the main result.

| Run | Direction | `predict_bleu-4` | ROUGE-1 | ROUGE-2 | ROUGE-L | Runtime |
|---|---|---:|---:|---:|---:|---:|
| Fine-tuned | Egyptian Arabic → English | 73.09 | 77.44 | 59.42 | 75.23 | 2:09:48 |
| Base | Egyptian Arabic → English | 44.70 | 50.34 | 28.26 | 46.06 | 2:53:36 |

> For final reported numbers, update these tables after each new run and consider evaluating the full test set instead of only 3000 samples.

---

## 7. Project structure

```text
configs/
  config.py          # reads settings from .env
  train.yaml         # LoRA training config
  eval_en2ar.yaml    # English -> Egyptian Arabic prediction
  eval_ar2en.yaml    # Egyptian Arabic -> English prediction

src/
  prepare.py         # builds train/val/test JSON files
  data.py            # cleaning, deduplication, splitting
  scheme.py          # converts pairs to Alpaca rows
  models.py          # tokenizer helpers
  evaluate.py        # BLEU and chrF++ scoring

data/
  dataset_info.json  # LLaMA-Factory dataset registry

dvc.yaml             # optional DVC pipeline
tests/               # pytest tests
```

---

## 8. Tests and linting

```bash
pytest -v
ruff check .
ruff format --check .
```

CI runs the same lint and test checks on pushes and pull requests to `main` / `master`.
