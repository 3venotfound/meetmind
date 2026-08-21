from fastapi import APIRouter

from app.api.meetings import router as meetings_router
from app.api.projects import router as projects_router
from app.api.results import router as results_router


api_router = APIRouter(prefix="/api")
api_router.include_router(projects_router)
api_router.include_router(meetings_router)
api_router.include_router(results_router)
