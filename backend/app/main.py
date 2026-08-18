from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import Settings, get_settings
from app.database import Database
from app.integrations import AIAdapter, CVAdapter
from app.repositories import Repository
from app.storage import RecordingStorage


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    database = Database(app_settings.resolved_database_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        app_settings.resolved_storage_root.mkdir(parents=True, exist_ok=True)
        database.initialize()
        application.state.settings = app_settings
        application.state.database = database
        application.state.repository = Repository(database)
        application.state.recording_storage = RecordingStorage(
            storage_root=app_settings.resolved_storage_root,
            max_upload_size_bytes=app_settings.max_upload_size_bytes,
        )
        application.state.ai_adapter = AIAdapter(
            storage_root=app_settings.resolved_storage_root,
            api_key=app_settings.gemini_api_key,
            python_executable=app_settings.ai_python_executable,
            timeout_seconds=app_settings.ai_timeout_seconds,
        )
        application.state.cv_adapter = CVAdapter(
            storage_root=app_settings.resolved_storage_root,
            python_executable=app_settings.cv_python_executable,
            timeout_seconds=app_settings.cv_timeout_seconds,
        )
        yield

    application = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    origins = app_settings.cors_origin_list
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router)

    return application


app = create_app()
