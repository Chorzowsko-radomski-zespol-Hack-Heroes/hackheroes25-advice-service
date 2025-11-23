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

# Niestandardowe middleware dla zezwalania na każdy localhost z dowolnym portem
class AllowAnyLocalhostMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("Origin")
        if origin and re.match(r"^http://localhost(:\d+)?$", origin):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
        return await call_next(request)

# Dodaj niestandardowe middleware
app.add_middleware(AllowAnyLocalhostMiddleware)

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
    return {"status": "ok"}
