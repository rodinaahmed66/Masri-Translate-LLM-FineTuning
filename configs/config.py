from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

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


def get_settings():
    return Settings()
