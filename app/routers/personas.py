from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from app.integrations.supabase import create_supabase_async_client
from app.repositories.user_persona_repository import UserPersonaProvider
from app.services.advice_service import build_user_persona_provider

router = APIRouter(prefix="/personas", tags=["personas"])


def get_persona_repository() -> UserPersonaProvider:
    client = create_supabase_async_client()
    return build_user_persona_provider(client)


@router.get("", summary="Get user persona text")
async def get_persona(
    user_id: str = Query(..., description="User ID"),
    persona_type: str = Query(
        "psychology", description="Type of persona: psychology, vocational, or tests"),
    repo: UserPersonaProvider = Depends(get_persona_repository),
) -> dict:
    """Get the persona text for a user."""
    if persona_type == "psychology" or persona_type == "vocational":
        persona_text = await repo.get_persona_by_type(user_id, persona_type)
    else:
        # For backward compatibility with "tests" or other types
        persona_text = await repo.get_persona(user_id)

    if persona_text is None:
        raise HTTPException(
            status_code=404, detail=f"Persona not found for user (type: {persona_type})")
    return {"user_id": user_id, "persona_type": persona_type, "persona_text": persona_text}
