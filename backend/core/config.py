from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "AI Tutor"
    admin_email: str
    database_url: str
    secret_key: str
    allowed_hosts: list = ["*"]
    debug: bool = False

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()