from pydantic import BaseModel, ValidationError, EmailStr
from datetime import date
from typing import Optional, Literal
from passlib.context import CryptContext
from .db import get_pool

class User(BaseModel):
  user_first_name: str
  user_middle_name: str
  user_last_name: str
  user_name: str
  email: EmailStr
  date_of_birth: date
  password: str
  role: Literal["user", "staff"]


INSERT_USER = f"""
INSERT INTO "user" (
    user_first_name,
    user_middle_name,
    user_last_name,
    user_name,
    email,
    date_of_birth,
    password,
    role
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING user_id, created_at;
"""
pwd_context = pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def get_password_hash(password):
    return pwd_context.hash(password)

async def signup(user: User):
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            INSERT_USER,
            user.user_first_name,
            user.user_middle_name,
            user.user_last_name,
            user.user_name,
            user.email,
            user.date_of_birth,
            get_password_hash(user.password), #hashed password
            user.role
        )
