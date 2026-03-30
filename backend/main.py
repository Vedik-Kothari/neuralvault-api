# =====================================================
# main.py
# FastAPI application entry point.
# This is the file you run to start the server.
#
# It wires together:
# - CORS middleware (allow Streamlit to call the API)
# - Routers (auth, documents)
# - Global exception handlers
# - Health check endpoint
# =====================================================

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import time
import sys

from .core.config import settings
from .routers import auth, documents, ingestion, query, rag


# ---- Logging Setup ----
# Loguru is simpler than Python's logging module
# Logs format: time | level | message
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG" if not settings.is_production else "INFO",
    colorize=True,
)


# ---- App Initialization ----
app = FastAPI(
    title="Enterprise Knowledge Base API",
    description=(
        "Secure RAG system with RBAC. "
        "All endpoints require JWT authentication except /health."
    ),
    version="1.0.0",
    # Disable docs in production (security best practice)
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None,
)


# ---- CORS Middleware ----
# CORS = Cross-Origin Resource Sharing
# Allows the Streamlit frontend (port 8501) to call FastAPI (port 8000)
# Without this, the browser blocks cross-origin requests.
# Remove the old CORSMiddleware and replace with this:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

from fastapi import Request
from fastapi.responses import Response

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return Response(
        content="OK",
        headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


# ---- Request Logging Middleware ----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs every request with timing.
    Useful for debugging and monitoring.
    """
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )
    return response


# ---- Global Exception Handler ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exception and returns a clean error response.
    Without this, unhandled errors expose stack traces to the client —
    a security risk in production.
    """
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again."}
    )


# ---- Routers ----
# Each router handles a group of related endpoints
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(ingestion.router) 
app.include_router(query.router)
app.include_router(rag.router)
# More routers added in Phase 3, 4, 5


# ---- Health Check ----
@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint.
    Used by deployment platforms to verify the app is running.
    No authentication required.
    """
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": "1.0.0"
    }


# ---- Startup Event ----
@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("Enterprise Knowledge Base API starting...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Docs available at: /docs")
    logger.info("=" * 50)