from src.scheme import to_alpaca

EN2AR = "to arabic"
AR2EN = "to english"


def test_alpaca_both_directions():
    ds = {"Arabic": ["ازيك"], "English": ["hello"]}
    by_input = {r["input"]: r for r in to_alpaca(ds, EN2AR, AR2EN)}
    assert by_input["hello"] == {
        "instruction": EN2AR,
        "input": "hello",
        "output": "ازيك",
    }
    assert by_input["ازيك"] == {
        "instruction": AR2EN,
        "input": "ازيك",
        "output": "hello",
    }


def test_alpaca_multiple():
    ds = {"Arabic": ["ازيك", "عامل ايه"], "English": ["hello", "how are you"]}
    rows = to_alpaca(ds, EN2AR, AR2EN)
    assert len(rows) == 4
    assert {r["input"] for r in rows} == {"hello", "how are you", "ازيك", "عامل ايه"}
    assert sum(r["instruction"] == EN2AR for r in rows) == 2
    assert sum(r["instruction"] == AR2EN for r in rows) == 2


def test_alpaca_empty():
    assert to_alpaca({"Arabic": [], "English": []}, EN2AR, AR2EN) == []


def test_alpaca_is_deterministic():
    ds = {
        "Arabic": [str(i) for i in range(20)],
        "English": [f"e{i}" for i in range(20)],
    }
    assert to_alpaca(ds, EN2AR, AR2EN) == to_alpaca(ds, EN2AR, AR2EN)
