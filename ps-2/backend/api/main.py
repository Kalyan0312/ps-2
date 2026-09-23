"""
Main FastAPI Application Entrypoint.

Serves REST endpoints, WebSocket telemetry, and the live dashboard UI.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.session import session_manager
from backend.api.routes.health import router as health_router
from backend.api.routes.streaming import router as stream_router
from backend.api.routes.replay import router as replay_router
from backend.api.routes.mapping import router as map_router
from backend.api.routes.metrics import router as metrics_router
from backend.api.websocket.streaming import router as ws_router

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring background tasks terminate gracefully."""
    yield
    # Cleanup on server shutdown
    if session_manager.is_streaming:
        await session_manager.stop_stream()


app = FastAPI(
    title="Adaptive Variable-Resolution 2.5D LiDAR Mapping",
    description="Real-time multi-resolution 2.5D elevation grid mapping with O(1) queries and temporal change tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API & WebSocket Routers
app.include_router(health_router)
app.include_router(replay_router)
app.include_router(stream_router)
app.include_router(map_router)
app.include_router(metrics_router)
app.include_router(ws_router)

# Mount Static Files
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the integrated browser dashboard UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "Adaptive LiDAR Mapping API is running. Dashboard assets pending."}
