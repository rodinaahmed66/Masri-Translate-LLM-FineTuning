"""Run the LLM judge on both directions.  Usage: python -m src.judge fine"""

import json
import os
import random
import re
import sys
from collections import defaultdict

from deepeval.test_case import LLMTestCase

from configs import get_settings
from src.LLM_as_Judge import GroqJudge, build_metrics

USER_TURN = re.compile(r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>", re.S)


def main():
    s = get_settings()
    tag = sys.argv[1] if len(sys.argv) > 1 else "fine"  # only used in file names
    os.makedirs(s.JUDGE_OUTPUT_DIR, exist_ok=True)
    metrics = build_metrics(GroqJudge())
    results = defaultdict(lambda: defaultdict(list))

    scores_path = f"{s.JUDGE_OUTPUT_DIR}/judge_scores_{tag}.jsonl"
    with open(scores_path, "w", encoding="utf-8") as out:
        for direction, path in [
            ("en2ar", s.PREDICTIONS_AR),
            ("ar2en", s.PREDICTIONS_EN),
        ]:
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            picked = random.Random(s.SEED).sample(
                range(len(rows)), min(s.JUDGE_N, len(rows))
            )
            for k, i in enumerate(sorted(picked), 1):
                r = rows[i]
                m = USER_TURN.search(r["prompt"])
                case = LLMTestCase(
                    input=m.group(1) if m else r["prompt"],
                    actual_output=r["predict"].strip(),
                    expected_output=r["label"].strip(),
                )
                scores = {}
                for metric in metrics:
                    try:
                        metric.measure(case)
                        score10 = round(metric.score * 10, 1)  # GEval returns 0-1
                        scores[metric.name] = {
                            "score": score10,
                            "reason": metric.reason,
                        }
                        results[direction][metric.name].append(score10)
                    except (
                        Exception
                    ) as e:  # one bad judge reply should not kill the run
                        scores[metric.name] = {"score": None, "reason": f"error: {e}"}
                        print(f"  {metric.name} failed: {e}")
                row = {
                    "direction": direction,
                    "idx": i,
                    "input": case.input,
                    "prediction": case.actual_output,
                    "reference": case.expected_output,
                    "scores": scores,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{tag}] {direction} {k}/{len(picked)}")

    summary = {
        d: {c: round(sum(v) / len(v), 2) for c, v in cs.items()}
        for d, cs in results.items()
    }
    with open(
        f"{s.JUDGE_OUTPUT_DIR}/judge_summary_{tag}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
