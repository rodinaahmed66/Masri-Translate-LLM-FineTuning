import json

from src.evaluate import arnorm, load_data, write_metrics


def test_load_data_skips_blank_lines(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"predict": " a ", "label": " b "}) + "\n\n",
        encoding="utf-8",
    )
    assert load_data(path) == (["a"], ["b"])


def test_arnorm():
    assert arnorm("إزّيكـ") == arnorm("ازيك")


def test_write_metrics(tmp_path):
    path = tmp_path / "preds.jsonl"
    write_metrics(str(path), 12.5, 34.5)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {"bleu": 12.5, "chrf++": 34.5}
