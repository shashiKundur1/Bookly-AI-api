from pydantic import BaseModel, EmailStr

from app.schemas.user import Name, Password, UserOut


class RegisterRequest(BaseModel):
    email: EmailStr
    password: Password
    name: Name


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
