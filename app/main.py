import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers.advice import router as advice_router
from app.routers.tests import router as tests_router
from app.routers.personas import router as personas_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Advice API")

app.include_router(advice_router)
app.include_router(tests_router)
app.include_router(personas_router)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
