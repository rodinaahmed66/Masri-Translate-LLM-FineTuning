from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    DATA_NAME: str
    TEST_SIZE: float 
    BASE_MODEL_ID:str


def get_settings():
    return Settings()