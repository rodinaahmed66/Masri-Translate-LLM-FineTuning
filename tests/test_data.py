import pytest
from datasets import Dataset

from src.data import DataLoader


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"English": "hello", "Arabic": "ازيك"}, True),  # normal
        ({"English": "hello", "Arabic": "ازيك OK"}, True),  # few latin letters ok
        ({"English": "", "Arabic": "ازيك"}, False),  # empty English
        ({"English": "hello", "Arabic": ""}, False),  # empty Arabic
        ({"English": "hi", "Arabic": "hello there"}, False),  # no Arabic at all
        ({"English": "hi", "Arabic": "ا hello world"}, False),  # more latin than Arabic
        ({"English": "hi", "Arabic": "aم"}, True),  # equal latin and arabic
    ],
)
def test_valid(row, expected):
    assert DataLoader.is_valid(row) is expected


def test_norm():
    assert DataLoader.norm("  hello   world  ") == "hello world"
    assert DataLoader.norm("foo\t\nbar") == "foo bar"


def test_duplicate():
    assert DataLoader.is_duplicate({"English": ["a", "b", "a"]}) == [0, 1]


def test_duplicate_unique():
    assert DataLoader.is_duplicate({"English": ["a", "b", "c"]}) == [0, 1, 2]


def test_duplicate_all_same():
    assert DataLoader.is_duplicate({"English": ["x", "x", "x"]}) == [0]


def test_prepare():
    row = {"Arabic": "[music] ازيك", "English": " Hello   World "}
    assert DataLoader("x").data_prepare(row) == {
        "Arabic": "ازيك",
        "English": "hello world",
    }


def test_prepare_multiple_sound_tags():
    row = {"Arabic": "[music] ازيك [sfx]", "English": "[noise] Hello   World [echo]"}
    assert DataLoader("x").data_prepare(row) == {
        "Arabic": "ازيك",
        "English": "hello world",
    }


def test_split():
    ds = Dataset.from_dict(
        {"English": [str(i) for i in range(100)], "Arabic": ["ا"] * 100}
    )
    splits = DataLoader("x", 0.1).split(ds)
    assert len(splits["test"]) == 10
