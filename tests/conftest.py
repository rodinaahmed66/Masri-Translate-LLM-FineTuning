import pytest


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {
        "DATA_NAME": "x",
        "TEST_SIZE": "0.1",
        "BASE_MODEL_ID": "x",
        "TOKENIZER_MAX_LENGTH": "512",
        "WANDB_API_KEY": "x",
        "PROJECT_NAME": "x",
        "HF_TOKEN": "x",
        "VAL_SIZE": "0.03",
        "PREDICTIONS_AR": "saves/exp10/eval_en2ar/generated_predictions.jsonl",
        "PREDICTIONS_EN": "saves/exp10/eval_ar2en/generated_predictions.jsonl",
    }.items():
        monkeypatch.setenv(k, v)
