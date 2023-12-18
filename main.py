import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from PromptKernel.PromptKernel import PromptKernel
from PromptKernel.Types.UserPrompts import UserPrompts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/")
def root(prompt: UserPrompts):
    kernel = PromptKernel()
    response = kernel.handle_prompt(prompt)
    return {'response': response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
