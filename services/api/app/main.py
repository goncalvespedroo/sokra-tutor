from fastapi import FastAPI

app = FastAPI(
    title="Sokra API",
    description="Sokra's main Backend",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "sokra-api",
    }
