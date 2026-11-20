import json

from configs.config import get_settings
from src.data import DataLoader
from src.models import Tokenizer
from src.scheme import INSTR_AR2EN, INSTR_EN2AR, to_alpaca, to_alpaca_bidirectional


def add_length(tokenizer: Tokenizer, row: dict) -> dict:
    """Length of the longer of the two prompt directions for a sentence pair."""
    en2ar = [
        tokenizer.construct_prompt("user", f"{INSTR_EN2AR}\n\n{row['English']}"),
        tokenizer.construct_prompt("assistant", row["Arabic"]),
    ]
    ar2en = [
        tokenizer.construct_prompt("user", f"{INSTR_AR2EN}\n\n{row['Arabic']}"),
        tokenizer.construct_prompt("assistant", row["English"]),
    ]
    return {
        "length": max(
            tokenizer.count_tokens(en2ar),
            tokenizer.count_tokens(ar2en),
        )
    }


def write_json(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    settings = get_settings()
    tokenizer = Tokenizer()

    loader = DataLoader(settings.DATA_NAME, settings.TEST_SIZE)
    data = loader.data_load()["train"]
    data = data.map(loader.data_prepare)
    data = data.select(DataLoader.is_duplicate(data))
    data = data.filter(DataLoader.is_valid)

    data = data.map(lambda row: add_length(tokenizer, row))
    data = data.filter(lambda row: row["length"] <= settings.TOKENIZER_MAX_LENGTH)
    data = data.remove_columns("length")

    splits = loader.split(data)

    write_json("data/train.json", to_alpaca_bidirectional(splits["train"]))
    write_json("data/test_en2ar.json", to_alpaca(splits["test"], INSTR_EN2AR))
    write_json("data/test_ar2en.json", to_alpaca(splits["test"], INSTR_AR2EN))


if __name__ == "__main__":
    main()
