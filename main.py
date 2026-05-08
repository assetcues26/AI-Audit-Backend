from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.audit import router as audit_router
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="Physical Asset Audit System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit_router, prefix="/api/v1/audit")

@app.get("/")
async def root():
    return {"message": "Physical Asset Audit System API is running"}
