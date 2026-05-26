from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.observability.metrics import setup_metrics
from app.observability.middleware import RequestContextMiddleware
from app.services.dependencies import build_container


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.container = build_container(settings)
    yield
    await app.state.container.shutdown()


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    app = FastAPI(
        title="Resume Analyzer API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/en", include_in_schema=False)
    @app.get("/en/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.include_router(router)
    setup_metrics(app)
    register_exception_handlers(app)
    return app


app = create_app()
