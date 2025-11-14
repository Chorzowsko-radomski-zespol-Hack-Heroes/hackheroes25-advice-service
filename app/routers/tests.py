from fastapi import APIRouter, Depends, HTTPException

from app.models.tests import PsychologyTestRequest, TestSubmissionResponse, VocationalTestRequest
from app.services.test_service import TestProcessingService, build_test_processing_service

router = APIRouter(prefix="/tests", tags=["tests"])


def get_test_service() -> TestProcessingService:
    return _TEST_SERVICE


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


_TEST_SERVICE = build_test_processing_service()

