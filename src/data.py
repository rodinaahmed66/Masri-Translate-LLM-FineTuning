import re
from typing import List
from datasets import Dataset, DatasetDict, load_dataset

AR = re.compile(r"[\u0600-\u06FF]")
LAT = re.compile(r"[A-Za-z]")
SOUND_TAGS = re.compile(r"\[[^\]\n]{1,30}\]")


class DataLoader:
    def __init__(self, data_name: str, test_size: float = 0.1):
        self.data_name = data_name
        self.test_size = test_size

    def data_load(self) -> DatasetDict:
        return load_dataset(self.data_name)

    @staticmethod
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def data_prepare(self, dataset: dict) -> dict:
        """Use with ds.map(): clean the text."""
        ar = self.norm(SOUND_TAGS.sub(" ", dataset["Arabic"]))
        en = self.norm(SOUND_TAGS.sub(" ", dataset["English"])).lower()
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
        seen, keep = set(), []
        for i, en in enumerate(dataset["English"]):
            if en not in seen:
                seen.add(en)
                keep.append(i)
        return keep

    def split(self, dataset: Dataset, seed: int = 42) -> DatasetDict:
        return dataset.train_test_split(test_size=self.test_size, seed=seed)
