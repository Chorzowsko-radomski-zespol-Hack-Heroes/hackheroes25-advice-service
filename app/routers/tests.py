from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.tests import (
    PsychologyTestRequest,
    PsychologyTestResultsResponse,
    TestSubmissionResponse,
    VocationalTestRequest,
    VocationalTestResultsResponse,
)
from app.services.test_service import TestProcessingService, build_test_processing_service

router = APIRouter(prefix="/tests", tags=["tests"])


def get_test_service() -> TestProcessingService:
    return build_test_processing_service()


@router.post(
    "/psychology",
    response_model=TestSubmissionResponse,
    summary="Submit psychology test answers",
)
async def submit_psychology_test(
    payload: PsychologyTestRequest,
    service: TestProcessingService = Depends(get_test_service),
) -> TestSubmissionResponse:
    try:
        return await service.submit_psychology_test(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/vocation",
    response_model=TestSubmissionResponse,
    summary="Submit vocational test answers",
)
async def submit_vocational_test(
    payload: VocationalTestRequest,
    service: TestProcessingService = Depends(get_test_service),
) -> TestSubmissionResponse:
    try:
        return await service.submit_vocational_test(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/psychology",
    response_model=PsychologyTestResultsResponse,
    summary="Get psychology test results for a user",
)
async def get_psychology_test_results(
    user_id: str = Query(..., description="User ID to get test results for"),
    service: TestProcessingService = Depends(get_test_service),
) -> PsychologyTestResultsResponse:
    results = await service.get_psychology_test_results(user_id)
    if not results:
        raise HTTPException(
            status_code=404, detail=f"Test not found"
        )
    return PsychologyTestResultsResponse(**results)


@router.get(
    "/vocation",
    response_model=VocationalTestResultsResponse,
    summary="Get vocational test results for a user",
)
async def get_vocational_test_results(
    user_id: str = Query(..., description="User ID to get test results for"),
    service: TestProcessingService = Depends(get_test_service),
) -> VocationalTestResultsResponse:
    results = await service.get_vocational_test_results(user_id)
    if not results:
        raise HTTPException(
            status_code=404, detail=f"Test not found"
        )
    return VocationalTestResultsResponse(**results)
