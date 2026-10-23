import json
import wandb
from huggingface_hub import login
from configs.config import get_settings
from src.data import DataLoader
from src.formatter import to_alpaca

settings = get_settings()

loader = DataLoader(settings.DATA_NAME, settings.TEST_SIZE)
data = loader.data_load()["train"]
data = data.map(loader.data_prepare)
data = data.filter(DataLoader.is_valid)
data = data.select(DataLoader.is_duplicate(data))
splits = loader.split(data)

for name in ("train", "test"):
    with open(f"data/{name}.json", "w", encoding="utf-8") as f:
        json.dump(to_alpaca(splits[name]), f, ensure_ascii=False, indent=2)

wandb.login(key=settings.WANDB_API_KEY)
login(token=settings.HF_TOKEN)
