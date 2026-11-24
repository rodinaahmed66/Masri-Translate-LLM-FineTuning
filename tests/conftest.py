import pytest


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k, v in {
        "DATA_NAME": "x",
        "BASE_MODEL_ID": "x",
        "TOKENIZER_MAX_LENGTH": "512",
        "WANDB_API_KEY": "x",
        "PROJECT_NAME": "x",
        "HF_TOKEN": "x",
        "VAL_SIZE": "0.03",
        "TEST_SIZE": "0.1",
        "PREDICTIONS_AR": "saves/exp10/eval_en2ar/generated_predictions.jsonl",
        "PREDICTIONS_EN": "saves/exp10/eval_ar2en/generated_predictions.jsonl",
        "GROQ_API_KEY": "x",
        "GROQ_BASE_URL": "https://api.groq.com/openai/v1",
        "JUDGE_MODEL": "x",
        "JUDGE_N": "5",
        "JUDGE_TEMPERATURE": "0",
        "JUDGE_MAX_TOKENS": "2048",
        "JUDGE_OUTPUT_DIR": "saves/judge",
        "MAX_RETRIES": "2",
        "CRITERIA": '["meaning","fluency","dialect","overall"]',
        "SEED": "42",
    }.items():
        monkeypatch.setenv(k, v)
