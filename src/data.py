import re
from datasets import Dataset, DatasetDict, load_dataset

AR = re.compile(r"[\u0600-\u06FF]")
LAT = re.compile(r"[A-Za-z]")


class DataLoader:
    def __init__(self, data_name: str, test_size: float = 0.1):
        self.data_name = data_name
        self.test_size = test_size

    def data_load(self) -> DatasetDict:
        return load_dataset(self.data_name)

    @staticmethod
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def data_prepare(self, example: dict) -> bool:
        ar = self.norm(example["Arabic"])
        en = self.norm(example["English"])
        if not ar or not en:
            return False
        if len(AR.findall(ar)) < 0.5 * len(ar.replace(" ", "")):
            return False
        if len(LAT.findall(en)) < 0.5 * len(en.replace(" ", "")):
            return False
        ratio = len(ar) / max(len(en), 1)
        if ratio < 0.3 or ratio > 3:
            return False
        if len(ar) > 600 or len(en) > 600:
            return False
        return True

    def split(self, ds: Dataset, seed: int = 42) -> dict[str, Dataset]:
        return ds.train_test_split(test_size=self.test_size, seed=seed)