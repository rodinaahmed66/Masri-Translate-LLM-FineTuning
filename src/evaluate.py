import json
import os
import re

import sacrebleu

PREDICTIONS_EN2AR = "saves/exp1/eval_en2ar/generated_predictions.jsonl"
PREDICTIONS_AR2EN = "saves/exp1/eval_ar2en/generated_predictions.jsonl"
REPORT = "saves/exp1/eval/metrics.json"


def load_data(filepath):
    preds, refs = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds.append(row["predict"].strip())
            refs.append(row["label"].strip())
    return preds, refs


def arnorm(s):
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    s = re.sub("[إأآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return s


def scores(filepath: str, normalize_arabic: bool = False):
    preds, refs = load_data(filepath)
    if normalize_arabic:
        preds = [arnorm(p) for p in preds]
        refs = [arnorm(r) for r in refs]
    tokenize = "intl" if normalize_arabic else "13a"
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize=tokenize).score
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    return bleu, chrf


def main():
    bleu_ar, chrf_ar = scores(PREDICTIONS_EN2AR, normalize_arabic=True)
    bleu_en, chrf_en = scores(PREDICTIONS_AR2EN)

    print("=" * 45)
    print(" 🇪🇬 English -> Egyptian Arabic Evaluation:")
    print(f"  BLEU   : {bleu_ar:.2f}")
    print(f"  chrF++ : {chrf_ar:.2f}")
    print(" 🇬🇧 Egyptian Arabic -> English Evaluation:")
    print(f"  BLEU   : {bleu_en:.2f}")
    print(f"  chrF++ : {chrf_en:.2f}")
    print("=" * 45)

    report = {
        "en2ar": {"bleu": bleu_ar, "chrf++": chrf_ar},
        "ar2en": {"bleu": bleu_en, "chrf++": chrf_en},
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
