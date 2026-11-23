import json
import os

from configs.config import get_settings
from src.data import DataLoader
from src.models import Tokenizer
from src.scheme import to_alpaca

INSTR_EN2AR = "Translate the following English text into Egyptian Arabic"
INSTR_AR2EN = "Translate the following Egyptian Arabic text into English"


def main():
    settings = get_settings()
    tokenizer = Tokenizer()

    loader = DataLoader(settings.DATA_NAME)
    data = loader.data_load()["train"]
    data = data.map(loader.data_prepare)
    data = data.select(loader.is_duplicate(data))
    data = data.filter(loader.is_valid)

    data = data.filter(
        lambda row: tokenizer.count_tokens(
            [
                {"role": "user", "content": f"{INSTR_EN2AR}\n\n{row['English']}"},
                {"role": "assistant", "content": row["Arabic"]},
            ]
        )
        <= settings.TOKENIZER_MAX_LENGTH
    )

    train_test = loader.split(data, settings.TEST_SIZE)
    test_ds = train_test["test"]
    train_val = loader.split(train_test["train"], settings.VAL_SIZE)
    train_ds, val_ds = train_val["train"], train_val["test"]

    train_rows = to_alpaca(train_ds, INSTR_EN2AR, INSTR_AR2EN)
    val_rows = to_alpaca(val_ds, INSTR_EN2AR, INSTR_AR2EN)

    test_en2ar = [
        {"instruction": INSTR_EN2AR, "input": en, "output": ar}
        for en, ar in zip(test_ds["English"], test_ds["Arabic"])
    ]
    test_ar2en = [
        {"instruction": INSTR_AR2EN, "input": ar, "output": en}
        for en, ar in zip(test_ds["English"], test_ds["Arabic"])
    ]

    os.makedirs("data", exist_ok=True)
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)

    with open("data/test_en2ar.json", "w", encoding="utf-8") as f:
        json.dump(test_en2ar, f, ensure_ascii=False, indent=2)

    with open("data/test_ar2en.json", "w", encoding="utf-8") as f:
        json.dump(test_ar2en, f, ensure_ascii=False, indent=2)

    with open("data/val.json", "w", encoding="utf-8") as f:
        json.dump(val_rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
