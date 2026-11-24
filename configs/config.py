from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CRITERIA = ["meaning", "fluency", "dialect", "overall"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATA_NAME: str
    BASE_MODEL_ID: str
    TOKENIZER_MAX_LENGTH: int
    WANDB_API_KEY: str
    PROJECT_NAME: str
    HF_TOKEN: str
    VAL_SIZE: float
    TEST_SIZE: float
    PREDICTIONS_AR: str
    PREDICTIONS_EN: str

    # LLM-as-judge. Optional so prepare/train/evaluate work without judge keys.
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    JUDGE_N: int = 50
    JUDGE_TEMPERATURE: float = 0.0
    JUDGE_MAX_TOKENS: int = 2048
    JUDGE_OUTPUT_DIR: str = "saves/judge"
    MAX_RETRIES: int = 6
    CRITERIA: list[str] = Field(default_factory=lambda: list(DEFAULT_CRITERIA))
    SEED: int = 42


def get_settings():
    return Settings()
