import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.integrations.supabase import create_supabase_async_client
from app.models.tests import PSYCHO_TRAITS, VOCATION_TRAITS
from app.repositories.job_demand_repository import (
    DemandLevel,
    DemandType,
    InMemoryJobDemandRepository,
    JobDemandRepository,
)
from app.repositories.test_repository import TestRepository
from code.career_adviser import get_jobs


router = APIRouter(prefix="/career_adviser", tags=["career_adviser"])
logger = logging.getLogger(__name__)


def _get_test_repository() -> TestRepository:
    client = create_supabase_async_client()
    return TestRepository(client)


def _map_demand_value_to_level(value: int) -> DemandLevel:
    """
    Mapuje wartość popytu (1-10) na poziom popytu.

    Args:
        value: Wartość popytu z zakresu 1-10

    Returns:
        Poziom popytu: veryLow, low, medium, high, veryHigh
    """
    if value <= 2:
        return "veryLow"
    elif value == 3:
        return "low"
    elif 4 <= value <= 6:
        return "medium"
    elif 7 <= value <= 8:
        return "high"
    else:  # 9-10
        return "veryHigh"


def _load_job_demands_from_files() -> dict[str, dict[DemandType, DemandLevel]]:
    """
    Ładuje dane popytu zawodów z plików zawody.txt (current) i zawody5.txt (in5years).

    Returns:
        Słownik z danymi popytu w formacie:
        {
            "nazwa_zawodu": {
                "current": "high",
                "in5years": "medium"
            },
            ...
        }
    """
    base_path = Path(__file__).parent.parent.parent
    current_file = base_path / "data" / "inout" / "zawody.txt"
    in5years_file = base_path / "data" / "inout" / "zawody5.txt"

    demands: dict[str, dict[DemandType, DemandLevel]] = {}

    # Wczytaj popyt obecny
    if current_file.exists():
        with open(current_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    job_name = parts[0].strip()
                    try:
                        value = int(parts[1].strip())
                        level = _map_demand_value_to_level(value)
                        if job_name not in demands:
                            demands[job_name] = {}
                        demands[job_name]["current"] = level
                    except ValueError:
                        logger.warning(
                            "Invalid demand value in zawody.txt: %s", line)
    else:
        logger.warning("File not found: %s", current_file)

    # Wczytaj popyt za 5 lat
    if in5years_file.exists():
        with open(in5years_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) == 2:
                    job_name = parts[0].strip()
                    try:
                        value = int(parts[1].strip())
                        level = _map_demand_value_to_level(value)
                        if job_name not in demands:
                            demands[job_name] = {}
                        demands[job_name]["in5years"] = level
                    except ValueError:
                        logger.warning(
                            "Invalid demand value in zawody5.txt: %s", line)
    else:
        logger.warning("File not found: %s", in5years_file)

    logger.info("Loaded demand data for %d jobs", len(demands))
    return demands


def _get_job_demand_repository() -> JobDemandRepository:
    """Tworzy repozytorium popytu zawodów z prostą mapą in-memory, wczytując dane z plików."""
    demands = _load_job_demands_from_files()
    return InMemoryJobDemandRepository(job_demands=demands)


def _is_demand_at_least_high(demand_level: DemandLevel | None) -> bool:
    """
    Sprawdza czy poziom popytu jest przynajmniej 'high'.

    Args:
        demand_level: Poziom popytu do sprawdzenia

    Returns:
        True jeśli popyt jest 'high' lub 'veryHigh', False w przeciwnym razie
    """
    if demand_level is None:
        return False
    return demand_level in ("high", "veryHigh")


@router.get("/advice", summary="Get career advice")
async def get_career_adviser(
    user_id: str = Query(..., description="User ID"),
    wpep_mode: int = Query(
        1, description="WPEP mode (0=none, 1=1 year, 2=5 years)"),
    demand: Literal["current", "in5years"] | None = Query(
        None, description="Filtruj według popytu: 'current' (obecny) lub 'in5years' (za 5 lat)"
    ),
) -> dict:
    """Get the career advice for a user."""
    logger.info(
        "Career advice request received for user_id=%s, demand=%s", user_id, demand)
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
    logger.info("Psychology traits retrieved: %d traits",
                len(psychology_traits))

    # Pobierz cechy zawodowe
    logger.info("Fetching vocational traits for user_id=%s", user_id)
    vocational_traits = await repository.get_traits(user_id, "vocational")
    if not vocational_traits:
        logger.warning(
            "Vocational test results not found for user_id=%s", user_id)
        raise HTTPException(
            status_code=404,
            detail="Brak wyników testu zawodowego dla tego użytkownika"
        )
    logger.info("Vocational traits retrieved: %d traits",
                len(vocational_traits))

    # Przygotuj listę cech w odpowiedniej kolejności: najpierw 8 psychologicznych, potem zawodowe
    psychology_traits_list = [
        psychology_traits.get(trait, 0.0) for trait in PSYCHO_TRAITS
    ]
    vocational_traits_list = [
        vocational_traits.get(trait, 0.0) for trait in VOCATION_TRAITS
    ]

    pers_input = psychology_traits_list + vocational_traits_list
    logger.info("Prepared pers_input with %d psychology traits and %d vocational traits",
                len(psychology_traits_list), len(vocational_traits_list))

    # Pobierz top 10 zawodów z sieci neuronowej
    jobs, scores = get_jobs(pers_input, 0, 10)
    scores = [float(s) for s in scores]

    # LOG: pełne score'y
    for j, s in zip(jobs, scores):
        logger.info("TOP score: %s -> %.4f", j, s)

    # Zawsze zwracamy absolute_best_job (najlepszy bez uwzględnienia popytu)
    absolute_best_job = jobs[0] if jobs else None

    # Jeśli nie podano parametru demand, zwracamy standardową odpowiedź
    if demand is None:
        return {
            "user_id": user_id,
            "jobs": jobs,
            "scores": scores,
            "absolute_best_job": absolute_best_job,
            "job_with_demand": None,
        }

    # Filtruj według popytu
    job_demand_repository = _get_job_demand_repository()
    demand_type: DemandType = demand

    # Bierzemy top 5 zawodów
    top_5_jobs = jobs[:5] if len(jobs) >= 5 else jobs
    top_5_scores = scores[:5] if len(scores) >= 5 else scores

    # Szukamy pierwszego z top 5, który ma demand przynajmniej "high"
    job_with_demand: str | None = None
    for job, score in zip(top_5_jobs, top_5_scores):
        job_demand = await job_demand_repository.get_demand(job, demand_type)
        logger.info("Job %s has demand %s (type: %s)",
                    job, job_demand, demand_type)
        if _is_demand_at_least_high(job_demand):
            job_with_demand = job
            logger.info("Found job with high demand: %s", job)
            break

    return {
        "user_id": user_id,
        "jobs": jobs,
        "scores": scores,
        "absolute_best_job": absolute_best_job,
        "job_with_demand": job_with_demand,
    }
