from configs.config import Settings


def _write_env(tmp_path, extra=""):
    path = tmp_path / ".env"
    path.write_text(
        "DATA_NAME=x\n"
        "BASE_MODEL_ID=x\n"
        "TOKENIZER_MAX_LENGTH=512\n"
        "WANDB_API_KEY=x\n"
        "PROJECT_NAME=x\n"
        "HF_TOKEN=x\n"
        "VAL_SIZE=0.03\n"
        "TEST_SIZE=0.1\n"
        "PREDICTIONS_AR=a.jsonl\n"
        "PREDICTIONS_EN=b.jsonl\n" + extra,
        encoding="utf-8",
    )
    return path


def test_extra_dotenv_key_is_ignored(tmp_path):
    env = _write_env(tmp_path, "UNKNOWN_KEY=hello\n")
    settings = Settings(_env_file=str(env))
    assert settings.PROJECT_NAME == "x"


def test_judge_settings_have_defaults(tmp_path, monkeypatch):
    for key in [
        "GROQ_API_KEY",
        "GROQ_BASE_URL",
        "JUDGE_MODEL",
        "JUDGE_N",
        "JUDGE_TEMPERATURE",
        "JUDGE_MAX_TOKENS",
        "JUDGE_OUTPUT_DIR",
        "MAX_RETRIES",
        "CRITERIA",
        "SEED",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=str(_write_env(tmp_path)))
    assert settings.GROQ_API_KEY == ""
    assert settings.JUDGE_N == 50
    assert settings.CRITERIA == ["meaning", "fluency", "dialect", "overall"]
