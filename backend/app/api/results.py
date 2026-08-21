from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.integrations.path_safety import UnsafePathError
from app.schemas import SearchRequest, SearchResponse


router = APIRouter(tags=["results"])


@router.get("/evidence/{evidence_id}/image", response_class=FileResponse)
def get_evidence_image(evidence_id: UUID, request: Request):
    relative_path = request.app.state.repository.get_evidence_path(str(evidence_id))
    if relative_path is None:
        raise HTTPException(status_code=404, detail="Evidence image not found")
    try:
        path = request.app.state.processing_service.evidence_store.resolve(relative_path)
    except (UnsafePathError, OSError, ValueError):
        raise HTTPException(status_code=404, detail="Evidence image not found") from None
    return FileResponse(path, media_type="image/jpeg", filename=f"{evidence_id}.jpg")


@router.post("/search", response_model=SearchResponse)
async def search_project(payload: SearchRequest, request: Request) -> dict:
    try:
        result = await request.app.state.processing_service.search(
            payload.project_id,
            payload.question,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail="Search failed") from error
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return result
