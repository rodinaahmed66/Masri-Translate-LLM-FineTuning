from src.scheme import to_alpaca


def test_alpaca():
    ds = {"Arabic": ["ازيك"], "English": ["hello"]}
    assert to_alpaca(ds, "T") == [{"instruction": "T", "input": "hello", "output": "ازيك"}]


def test_alpaca_multiple():
    ds = {"Arabic": ["ازيك", "عامل ايه"], "English": ["hello", "how are you"]}
    assert to_alpaca(ds, "Translate") == [
        {"instruction": "Translate", "input": "hello", "output": "ازيك"},
        {"instruction": "Translate", "input": "how are you", "output": "عامل ايه"},
    ]


def test_alpaca_empty():
    ds = {"Arabic": [], "English": []}
    assert to_alpaca(ds, "T") == []
