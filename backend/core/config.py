from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "AI Tutor"

    admin_email: str
    database_url: str

    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str


    secret_key: str
    algorithm: str = "HS256"
    allowed_hosts: list = ["*"]
    debug: bool = False

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()