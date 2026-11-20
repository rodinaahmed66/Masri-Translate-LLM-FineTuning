# Masri-Translate

Egyptian Arabic <-> English translation fine-tuning with LLaMA-Factory.

## Pipeline

The DVC pipeline (`dvc.yaml`) has four stages:

1. `prepare` — clean, dedupe and length-filter the dataset, then write:
   - `data/train.json` — both translation directions, interleaved and shuffled
   - `data/test_en2ar.json`, `data/test_ar2en.json` — direction-specific eval sets
2. `train` — LoRA SFT on `masri_mt_train`
3. `predict_en2ar` / `predict_ar2en` — generation with the LoRA adapter
4. `metrics` — BLEU and chrF++ for both directions, written to `saves/exp1/eval/metrics.json`

Run the whole pipeline with `dvc repro`.
