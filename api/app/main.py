from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.database import init_db
from app.routes import tryon, status
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Ensure storage directories exist
    for sub in ("inputs", "outputs", "training_pairs"):
        Path(settings.LOCAL_STORAGE_PATH, sub).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="VTON API",
    description="Virtual Try-On API — placeholder inference, async Celery jobs, auto training pair collection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tryon.router, prefix="/api/v1", tags=["tryon"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])

# Serve result images at /files/<relative-path>
storage_path = Path(settings.LOCAL_STORAGE_PATH)
storage_path.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(storage_path)), name="files")

# Serve frontend UI
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "vton-api"}
