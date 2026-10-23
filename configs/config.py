from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATA_NAME: str
    TEST_SIZE: float 
    BASE_MODEL_ID:str
    TOKENIZER_MAX_LENGTH:int
    WANDB_API_KEY:str
    PROJECT_NAME:str
    HF_TOKEN:str


def get_settings():
    return Settings()