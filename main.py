from fastapi import FastAPI

from .routers import strategy1_api

app = FastAPI()


app.include_router(strategy1_api.router)

@app.get("/")
async def root():
    return {"message": "Pong"}