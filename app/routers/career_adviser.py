import logging

from fastapi import APIRouter, HTTPException, Query

from app.integrations.supabase import create_supabase_async_client
from app.models.tests import PSYCHO_TRAITS, VOCATION_TRAITS
from app.repositories.test_repository import TestRepository
from code.career_adviser import get_jobs


router = APIRouter(prefix="/career_adviser", tags=["career_adviser"])
logger = logging.getLogger(__name__)


def _get_test_repository() -> TestRepository:
    client = create_supabase_async_client()
    return TestRepository(client)


@router.get("/advice", summary="Get career advice")
async def get_career_adviser(
    user_id: str = Query(..., description="User ID"),
    wpep_mode: int = Query(
        1, description="WPEP mode (0=none, 1=1 year, 2=5 years)"),
) -> dict:
    """Get the career advice for a user."""
    logger.info("Career advice request received for user_id=%s", user_id)
    repository = _get_test_repository()

    # Pobierz cechy psychologiczne (8 cech)
    logger.debug("Fetching psychology traits for user_id=%s", user_id)
    psychology_traits = await repository.get_traits(user_id, "psychology")
    if not psychology_traits:
        logger.warning(
            "Psychology test results not found for user_id=%s", user_id)
        raise HTTPException(
            status_code=404,
            detail="Brak wyników testu psychologicznego dla tego użytkownika"
        )
    logger.debug("Psychology traits retrieved: %d traits",
                 len(psychology_traits))

    # Pobierz cechy zawodowe
    logger.debug("Fetching vocational traits for user_id=%s", user_id)
    vocational_traits = await repository.get_traits(user_id, "vocational")
    if not vocational_traits:
        logger.warning(
            "Vocational test results not found for user_id=%s", user_id)
        raise HTTPException(
            status_code=404,
            detail="Brak wyników testu zawodowego dla tego użytkownika"
        )
    logger.debug("Vocational traits retrieved: %d traits",
                 len(vocational_traits))

    # Przygotuj listę cech w odpowiedniej kolejności: najpierw 8 psychologicznych, potem zawodowe
    psychology_traits_list = [
        psychology_traits.get(trait, 0.0) for trait in PSYCHO_TRAITS
    ]
    vocational_traits_list = [
        vocational_traits.get(trait, 0.0) for trait in VOCATION_TRAITS
    ]

    pers_input = psychology_traits_list + vocational_traits_list
    logger.debug("Prepared pers_input with %d psychology traits and %d vocational traits",
                 len(psychology_traits_list), len(vocational_traits_list))

    jobs, scores = get_jobs(pers_input, wpep_mode, 5)
    scores = [float(s) for s in scores]
    logger.info("Career advice generated for user_id=%s: top_job=%s score=%.2f",
                user_id, jobs[0] if jobs else "N/A", scores[0] if scores else 0.0)
    best_job = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)[
        0]
    # sort jobs by scores and return the best one

    return {"user_id": user_id, "jobs": jobs, "scores": scores, "best_job": best_job}
