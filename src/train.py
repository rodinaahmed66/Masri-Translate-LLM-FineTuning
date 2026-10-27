import json
import wandb
from huggingface_hub import login
from configs.config import get_settings
from src.data import DataLoader
from src.scheme import to_alpaca
from src.models import Tokenizer

tokenizer = Tokenizer()
settings = get_settings()

wandb.login(key=settings.WANDB_API_KEY)
wandb.init(project=settings.PROJECT_NAME)
login(token=settings.HF_TOKEN)


INSTR = "Translate the following English text into Egyptian Arabic."

loader = DataLoader(settings.DATA_NAME, settings.TEST_SIZE)
data = loader.data_load()["train"]
data = data.map(loader.data_prepare)
data = data.filter(DataLoader.is_valid)
data = data.select(DataLoader.is_duplicate(data))

message = [
    tokenizer.construct_prompt("user", f"{INSTR}\n\n{data["English"]}"),
    tokenizer.construct_prompt("assistant", data["Arabic"]),
]

tokens = tokenizer.tokenize(data)
data = data.filter(len(tokens["input_ids"]) <= settings.TOKENIZER_MAX_LENGTH)


splits = loader.split(data)

for name in ("train", "test"):
    with open(f"data/{name}.json", "w", encoding="utf-8") as f:
        json.dump(to_alpaca(splits[name], INSTR), f, ensure_ascii=False, indent=2)
