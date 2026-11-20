import json

import src.evaluate as evaluate


def test_load_data(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"predict": " ازيك ", "label": " عامل ايه "}) + "\n",
        encoding="utf-8",
    )
    assert evaluate.load_data(path) == (["ازيك"], ["عامل ايه"])


def test_arnorm():
    assert evaluate.arnorm("إزّيكْ") == "ازيك"
    assert evaluate.arnorm("عامل ايه") == "عامل ايه"


def test_main_writes_both_directions(tmp_path, monkeypatch):
    for direction in ("en2ar", "ar2en"):
        path = tmp_path / f"{direction}.jsonl"
        path.write_text(
            json.dumps({"predict": "ازيك", "label": "ازيك"}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            evaluate, f"PREDICTIONS_{direction.upper()}", str(path)
        )
    report_path = tmp_path / "metrics.json"
    monkeypatch.setattr(evaluate, "REPORT", str(report_path))

    evaluate.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"en2ar", "ar2en"}
    assert set(report["en2ar"]) == {"bleu", "chrf++"}
    assert set(report["ar2en"]) == {"bleu", "chrf++"}
