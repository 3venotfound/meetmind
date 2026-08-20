from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from app.repositories import Repository
from app.schemas import ProjectCreate, ProjectHistoryResponse, ProjectResponse


router = APIRouter(prefix="/projects", tags=["projects"])


def _repository(request: Request) -> Repository:
    return request.app.state.repository


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, request: Request) -> dict:
    return _repository(request).create_project(project)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: UUID, request: Request) -> dict:
    project = _repository(request).get_project(str(project_id))
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/{project_id}/history", response_model=ProjectHistoryResponse)
def get_project_history(project_id: UUID, request: Request) -> dict:
    history = _repository(request).get_project_history(str(project_id))
    if history is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project_id": project_id, "history": history}
