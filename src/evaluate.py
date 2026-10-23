import json
import re
import sacrebleu

PREDICTIONS = "saves/exp1/eval/generated_predictions.jsonl"
REPORT = "saves/exp1/eval/metrics.json"

preds, refs = [], []
with open(PREDICTIONS, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        preds.append(row["predict"].strip())
        refs.append(row["label"].strip())


def arnorm(s):
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)
    s = re.sub("[إأآ]", "ا", s).replace("ى", "ي")
    return s


preds = [arnorm(p) for p in preds]
refs = [arnorm(r) for r in refs]

bleu = sacrebleu.corpus_bleu(preds, [refs]).score
chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score

print("BLEU :", bleu)
print("chrF++:", chrf)

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump({"bleu": bleu, "chrf++": chrf}, f, indent=2)
