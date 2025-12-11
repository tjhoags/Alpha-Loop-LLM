from pydantic import BaseModel


class Prompt(BaseModel):
    id: str
    title: str
    content: str
    price: float = 0.0
    author: str = "anon"


