import json

from configs.config import get_settings
from src.data import DataLoader
from src.models import Tokenizer
from src.scheme import to_alpaca

INSTR = "Translate the following English text into Egyptian Arabic."


def add_length(tokenizer: Tokenizer, row: dict) -> dict:
    messages = [
        tokenizer.construct_prompt("user", f"{INSTR}\n\n{row['English']}"),
        tokenizer.construct_prompt("assistant", row["Arabic"]),
    ]
    return {"length": tokenizer.count_tokens(messages)}


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

    for name in ("train", "test"):
        with open(f"data/{name}.json", "w", encoding="utf-8") as f:
            json.dump(to_alpaca(splits[name], INSTR), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
