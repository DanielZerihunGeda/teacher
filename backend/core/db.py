import asyncpg
from config import Settings

setting = Settings()
from fastapi import FastAPI

pool: asyncpg.Pool = None

async def init_db():
    global pool
    pool = await asyncpg.create_pool(dsn=setting.database_url)

async def close_db():
    await pool.close()

def get_pool() -> asyncpg.Pool:
    return pool

def register_db_events(app: FastAPI):
    app.on_event("startup")(init_db)
    app.on_event("shutdown")(close_db)