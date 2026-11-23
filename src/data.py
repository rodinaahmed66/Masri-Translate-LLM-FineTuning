import re
import string
from typing import List

from datasets import Dataset, DatasetDict, load_dataset

AR = re.compile(r"[\u0600-\u06FF]")
LAT = re.compile(r"[A-Za-z]")
SOUND_TAGS = re.compile(r"\[[^\]\n]{1,30}\]")
PUNCT = re.compile(r"[\s" + re.escape(string.punctuation) + r"،؟؛]+")


class DataLoader:
    def __init__(self, data_name: str):
        self.data_name = data_name

    def data_load(self) -> DatasetDict:
        return load_dataset(self.data_name)

    @staticmethod
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def key_ar(text: str) -> str:
        """Normalized Arabic key used for deduplication."""
        text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
        text = re.sub("[إأآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
        return PUNCT.sub("", text)

    @staticmethod
    def key_en(text: str) -> str:
        """Normalized English key used for deduplication."""
        return PUNCT.sub("", text.lower())

    def data_prepare(self, dataset: dict) -> dict:
        """Use with ds.map(): clean the text."""
        ar = self.norm(SOUND_TAGS.sub(" ", dataset["Arabic"]))
        en = self.norm(SOUND_TAGS.sub(" ", dataset["English"]))
        return {"Arabic": ar, "English": en}

    @staticmethod
    def is_valid(dataset: dict) -> bool:
        en, ar = dataset["English"], dataset["Arabic"]
        if not en or not ar:
            return False
        n_ar = len(AR.findall(ar))
        if n_ar == 0 or len(LAT.findall(ar)) > n_ar:
            return False
        return True

    @staticmethod
    def is_duplicate(dataset) -> List:
        """Keep the first row per unique English *and* Arabic normalized key."""

        seen_en, seen_ar, keep = set(), set(), []
        for i, (en, ar) in enumerate(zip(dataset["English"], dataset["Arabic"])):
            key_en = DataLoader.key_en(en)
            key_ar = DataLoader.key_ar(ar) if ar is not None else None
            if key_en in seen_en or key_ar in seen_ar:
                continue

            seen_en.add(key_en)
            seen_ar.add(key_ar)
            keep.append(i)
        return keep

    def split(self, dataset: Dataset, test_size: float, seed: int = 42) -> DatasetDict:
        return dataset.train_test_split(test_size=test_size, seed=seed)
