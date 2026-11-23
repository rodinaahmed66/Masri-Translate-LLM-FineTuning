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
    ds = {"English": ["a", "b", "a"], "Arabic": ["ازيك", "عامل", "كويس"]}
    assert DataLoader.is_duplicate(ds) == [0, 1]


def test_duplicate_unique():
    ds = {"English": ["a", "b", "c"], "Arabic": ["ازيك", "عامل", "كويس"]}
    assert DataLoader.is_duplicate(ds) == [0, 1, 2]


def test_duplicate_all_same():
    ds = {"English": ["x", "x", "x"], "Arabic": ["ازيك", "عامل", "كويس"]}
    assert DataLoader.is_duplicate(ds) == [0]


def test_duplicate_by_english_key():
    # case and punctuation are ignored by key_en
    ds = {"English": ["Hello", "hello!"], "Arabic": ["ازيك", "عامل"]}
    assert DataLoader.is_duplicate(ds) == [0]


def test_duplicate_by_arabic_key():
    # tatweel is removed by key_ar, so both Arabic texts match
    ds = {"English": ["a", "b"], "Arabic": ["ازيك", "ازيكـ"]}
    assert DataLoader.is_duplicate(ds) == [0]


def test_keys():
    assert DataLoader.key_en(" Hello, World! ") == "helloworld"
    assert DataLoader.key_ar("اَزّيك") == DataLoader.key_ar("ازيك")


def test_prepare():
    row = {"Arabic": "[music] ازيك", "English": " Hello World "}
    assert DataLoader("x").data_prepare(row) == {
        "Arabic": "ازيك",
        "English": "Hello World",
    }


def test_prepare_multiple_sound_tags():
    row = {"Arabic": "[music] ازيك [sfx]", "English": "[noise] Hello World [echo]"}
    assert DataLoader("x").data_prepare(row) == {
        "Arabic": "ازيك",
        "English": "Hello World",
    }


def test_split():
    ds = Dataset.from_dict({"English": [str(i) for i in range(100)], "Arabic": ["ا"] * 100})
    splits = DataLoader("x").split(ds, 0.1)
    assert len(splits["test"]) == 10


def test_split_two_times():
    ds = Dataset.from_dict({"English": [str(i) for i in range(100)], "Arabic": ["ا"] * 100})
    loader = DataLoader("x")
    train_test = loader.split(ds, 0.1)
    train_val = loader.split(train_test["train"], 0.1)
    assert len(train_test["test"]) == 10
    assert len(train_val["test"]) == 9
    assert len(train_val["train"]) == 81


def test_split_is_reproducible():
    ds = Dataset.from_dict({"English": [str(i) for i in range(100)], "Arabic": ["ا"] * 100})
    a = DataLoader("x").split(ds, 0.1, seed=1)["test"]["English"]
    b = DataLoader("x").split(ds, 0.1, seed=1)["test"]["English"]
    assert a == b
