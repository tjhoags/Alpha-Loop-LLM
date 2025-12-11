from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Prompt Forge", version="0.1.0")


class Prompt(BaseModel):
    id: str
    title: str
    content: str
    price: float = 0.0
    author: str = "anon"


PROMPTS: List[Prompt] = []


@app.get("/health")
def health():
    return {"status": "ok", "service": "prompt_forge"}


@app.get("/prompts", response_model=List[Prompt])
def list_prompts():
    return PROMPTS


@app.post("/prompts", response_model=Prompt)
def create_prompt(prompt: Prompt):
    if any(p.id == prompt.id for p in PROMPTS):
        raise HTTPException(status_code=400, detail="Prompt ID already exists")
    PROMPTS.append(prompt)
    return prompt


@app.get("/prompts/{prompt_id}", response_model=Prompt)
def get_prompt(prompt_id: str):
    for prompt in PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise HTTPException(status_code=404, detail="Prompt not found")


