from datetime import datetime, timezone, timedelta
from typing import Literal

import os
import asyncpg
import httpx
import jwt
from fastapi import FastAPI, Depends, HTTPException, Query, Response
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from .db import get_pool
from .config import Settings

settings = Settings()

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

GOOGLE_CLIENT_ID = settings.google_client_id
GOOGLE_CLIENT_SECRET = settings.google_client_secret
GOOGLE_REDIRECT_URI = settings.google_redirect_uri
ALGORITHM = settings.algorithm
SECRET_KEY = settings.secret_key


class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    role: Literal["user", "staff"] = "user"


async def get_connection(pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        yield conn


INSERT_USER = """
INSERT INTO "user" (first_name, last_name, email, password, role)
VALUES ($1, $2, $3, $4, $5)
RETURNING user_id, created_at;
"""


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


async def get_password_hash(password: str):
    return pwd_context.hash(password)


async def check_user_exists(email: str, conn=Depends(get_connection)) -> bool:
    row = await conn.fetchrow('SELECT email FROM "user" WHERE email = $1;', email)
    return row is not None


async def create_user(email: str, first_name: str, last_name: str, password: str, conn=Depends(get_connection)):
    hashed_pw = await get_password_hash(password)
    return await conn.fetchrow(
        INSERT_USER, first_name, last_name, email, hashed_pw, "user"
    )


@app.get("/auth/google/callback")
async def google_callback(response: Response, code: str = Query(...)):
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        if token_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Google token exchange failed")

        tokens = token_res.json()
        id_info_res = await client.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={tokens['id_token']}"
        )
        id_info = id_info_res.json()

    if id_info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=400, detail="Invalid audience in token")

    email = id_info["email"]
    first_name = id_info.get("given_name", "")
    last_name = id_info.get("family_name", "")

    if not await check_user_exists(email):
        await create_user(email, first_name, last_name, os.urandom(16).hex())

    jwt_token = create_access_token({"sub": email})
    

    # Save token in HttpOnly cookie
    response.set_cookie(
          key="access_token",
          value=jwt_token,
          httponly=True,
          secure=True,
          samesite="strict"
    )


    return {"message":"Login Successful, JWT stored in cookie"}
