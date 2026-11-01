import json
import os
import re

import sacrebleu

PREDICTIONS = "saves/exp1/eval/generated_predictions.jsonl"
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


def main():
    preds, refs = load_data(PREDICTIONS)
    preds = [arnorm(p) for p in preds]
    refs = [arnorm(r) for r in refs]

    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl").score
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

    print("=" * 45)
    print(" 🇪🇬 English -> Egyptian Arabic Evaluation:")
    print(f"  BLEU   : {bleu:.2f}")
    print(f"  chrF++ : {chrf:.2f}")
    print("=" * 45)

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"bleu": bleu, "chrf++": chrf}, f, indent=2)


if __name__ == "__main__":
    main()
