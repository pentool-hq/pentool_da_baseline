# main.py

from fastapi import FastAPI
from routers import api

app = FastAPI(
    title="Pendle YT Timing Strategy Analyzer API",
    description="API for fetching data and predicting order arrival times based on transaction data.",
    version="1.0.0"
)

app.include_router(api.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
