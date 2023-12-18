from pydantic import BaseModel


class UserPrompts(BaseModel):
    prompt: str
    history: str
