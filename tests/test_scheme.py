from src.scheme import (
    INSTR_AR2EN,
    INSTR_EN2AR,
    to_alpaca,
    to_alpaca_bidirectional,
)


def _triples(rows):
    return {(r["instruction"], r["input"], r["output"]) for r in rows}


def test_alpaca():
    ds = {"Arabic": ["ازيك"], "English": ["hello"]}
    assert to_alpaca(ds, "T") == [
        {"instruction": "T", "input": "hello", "output": "ازيك"}
    ]


def test_alpaca_multiple():
    ds = {"Arabic": ["ازيك", "عامل ايه"], "English": ["hello", "how are you"]}
    assert to_alpaca(ds, "Translate") == [
        {"instruction": "Translate", "input": "hello", "output": "ازيك"},
        {"instruction": "Translate", "input": "how are you", "output": "عامل ايه"},
    ]


def test_alpaca_empty():
    ds = {"Arabic": [], "English": []}
    assert to_alpaca(ds, "T") == []


def test_alpaca_bidirectional():
    ds = {"Arabic": ["ازيك"], "English": ["hello"]}
    assert _triples(to_alpaca_bidirectional(ds)) == {
        (INSTR_EN2AR, "hello", "ازيك"),
        (INSTR_AR2EN, "ازيك", "hello"),
    }


def test_alpaca_bidirectional_multiple():
    ds = {"Arabic": ["ازيك", "عامل ايه"], "English": ["hello", "how are you"]}
    rows = to_alpaca_bidirectional(ds)
    assert len(rows) == 4
    assert _triples(rows) == {
        (INSTR_EN2AR, "hello", "ازيك"),
        (INSTR_AR2EN, "ازيك", "hello"),
        (INSTR_EN2AR, "how are you", "عامل ايه"),
        (INSTR_AR2EN, "عامل ايه", "how are you"),
    }


def test_alpaca_bidirectional_is_deterministic():
    ds = {"Arabic": ["ازيك", "عامل ايه"], "English": ["hello", "how are you"]}
    assert to_alpaca_bidirectional(ds) == to_alpaca_bidirectional(ds)


def test_alpaca_bidirectional_empty():
    assert to_alpaca_bidirectional({"Arabic": [], "English": []}) == []
