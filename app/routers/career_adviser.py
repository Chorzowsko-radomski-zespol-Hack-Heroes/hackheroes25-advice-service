

from fastapi import APIRouter, Query


router = APIRouter(prefix="/career_adviser", tags=["career_adviser"])


@router.get("advice", summary="Get career advice")
async def get_career_adviser(
    user_id: str = Query(..., description="User ID"),

) -> dict:
    """Get the career advice for a user."""
    return {"user_id": user_id, "job": "Hydraulik"}
