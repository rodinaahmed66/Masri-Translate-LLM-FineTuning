import json
import os
import re

import sacrebleu

from configs.config import get_settings

PREDICTIONS_ar = get_settings().PREDICTIONS_AR
PREDICTIONS_en = get_settings().PREDICTIONS_EN


def load_data(filepath):
    preds, refs = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            preds.append(d["predict"].strip())
            refs.append(d["label"].strip())
    return preds, refs


def arnorm(s):
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    s = re.sub("[إأآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return s


def write_metrics(prediction_path, bleu, chrf):
    path = os.path.join(os.path.dirname(prediction_path), "metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"bleu": bleu, "chrf++": chrf}, f, indent=2)


def main():
    # 1. English -> Egyptian Arabic
    p_ar, r_ar = load_data(PREDICTIONS_ar)
    p_ar_norm = [arnorm(p) for p in p_ar]
    r_ar_norm = [arnorm(r) for r in r_ar]

    bleu_ar = sacrebleu.corpus_bleu(p_ar_norm, [r_ar_norm], tokenize="intl").score
    chrf_ar = sacrebleu.corpus_chrf(p_ar_norm, [r_ar_norm], word_order=2).score

    print("=" * 45)
    print(" 🇪🇬 English -> Egyptian Arabic Evaluation:")
    print(f"  BLEU   : {bleu_ar:.2f}")
    print(f"  chrF++ : {chrf_ar:.2f}")

    # 2. Egyptian Arabic -> English
    p_en, r_en = load_data(PREDICTIONS_en)

    bleu_en = sacrebleu.corpus_bleu(p_en, [r_en]).score
    chrf_en = sacrebleu.corpus_chrf(p_en, [r_en], word_order=2).score

    print("=" * 45)
    print(" 🇬🇧 Egyptian Arabic -> English Evaluation:")
    print(f"  BLEU   : {bleu_en:.2f}")
    print(f"  chrF++ : {chrf_en:.2f}")
    print("=" * 45)

    write_metrics(PREDICTIONS_ar, bleu_ar, chrf_ar)
    write_metrics(PREDICTIONS_en, bleu_en, chrf_en)


if __name__ == "__main__":
    main()
