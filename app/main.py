import logging
import re
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.routers import career_adviser
from app.routers.advice import router as advice_router
from app.routers.tests import router as tests_router
from app.routers.personas import router as personas_router
from app.routers.career_adviser import router as career_adviser_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Advice API")

app.add_middleware(
    CORSMiddleware,
    # any port, forever
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    expose_headers=["*"],
    max_age=600,
)

app.include_router(advice_router)
app.include_router(tests_router)
app.include_router(personas_router)
app.include_router(career_adviser_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
@app.get("/health")
def health_check():
    """Health check endpoint for Fly.io and load balancers."""
@app.get("/health")
def health():
    return {"status": "ok"}
