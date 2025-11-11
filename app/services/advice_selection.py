from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol, Sequence

from app.models.advice import Advice, AdviceKind, AdviceRecommendation, AdviceRequestContext
from app.repositories.advice_repository import AdviceRepository
from app.repositories.category_repository import AdviceCategoryRepository


class AdviceCategoryClassifier(Protocol):
    async def infer_categories(self, user_message: str) -> Sequence[str]:
        raise NotImplementedError


class AdviceIntentDetector(Protocol):
    async def detect_preferred_kind(self, user_message: str) -> AdviceKind | None:
        raise NotImplementedError


class AdviceResponseGenerator(Protocol):
    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        raise NotImplementedError


class AdviceNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdviceSelectionResult:
    advice: Advice
    categories: Sequence[str]
    preferred_kind: AdviceKind | None


class AdviceSelectionPipeline:
    def __init__(
        self,
        advice_repository: AdviceRepository,
        category_repository: AdviceCategoryRepository,
        category_classifier: AdviceCategoryClassifier,
        intent_detector: AdviceIntentDetector,
        response_generator: AdviceResponseGenerator,
    ) -> None:
        self._advice_repository = advice_repository
        self._category_repository = category_repository
        self._category_classifier = category_classifier
        self._intent_detector = intent_detector
        self._response_generator = response_generator

    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        categories = await self._infer_categories(request.user_message)
        preferred_kind = await self._intent_detector.detect_preferred_kind(
            request.user_message
        )

        advice = await self._select_advice(preferred_kind, categories)
        if advice is None:
            raise AdviceNotFoundError(
                "No advice found for the given criteria.")

        chat_response = await self._response_generator.generate_response(
            advice=advice,
            request=request,
            categories=categories,
            preferred_kind=preferred_kind,
        )
        return AdviceRecommendation(advice=advice, chat_response=chat_response)

    async def _infer_categories(self, user_message: str) -> Sequence[str]:
        inferred = await self._category_classifier.infer_categories(user_message)
        if not inferred:
            return ()
        known_categories = await self._category_repository.get_all()
        known_set = {category.lower() for category in known_categories}
        unique_categories = []
        for category in inferred:
            normalized = category.lower()
            if normalized in known_set and normalized not in unique_categories:
                unique_categories.append(normalized)
        return tuple(unique_categories)

    async def _select_advice(
        self,
        preferred_kind: AdviceKind | None,
        categories: Sequence[str],
    ) -> Advice | None:
        candidates: Sequence[Advice] = ()

        if preferred_kind is not None:
            if categories:
                candidates = await self._advice_repository.get_by_kind_and_containing_any_category(
                    preferred_kind,
                    categories,
                )
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)

        if not candidates:
            if categories:
                all_advice = await self._advice_repository.get_all()
                candidates = tuple(
                    advice
                    for advice in all_advice
                    if self._contains_any_category(advice, categories)
                )

        if not candidates:
            candidates = await self._advice_repository.get_all()

        return self._rank_candidates(candidates, categories)

    @staticmethod
    def _contains_any_category(advice: Advice, categories: Sequence[str]) -> bool:
        advice_categories = {category.lower()
                             for category in advice.categories}
        for category in categories:
            if category.lower() in advice_categories:
                return True
        return False

    def _rank_candidates(
        self, candidates: Sequence[Advice], categories: Sequence[str]
    ) -> Advice | None:
        if not candidates:
            return None

        if not categories:
            return random.choice(tuple(candidates))

        category_set = {category.lower() for category in categories}

        scored_candidates: list[tuple[int, Advice]] = []
        for candidate in candidates:
            candidate_categories = {
                category.lower() for category in candidate.categories
            }
            overlap = len(candidate_categories & category_set)
            scored_candidates.append((overlap, candidate))

        grouped: dict[int, list[Advice]] = {}
        for score, candidate in scored_candidates:
            grouped.setdefault(score, []).append(candidate)

        for target_score in (3, 2, 1):
            if target_score in grouped:
                return random.choice(grouped[target_score])

        return None


class EchoAdviceResponseGenerator(AdviceResponseGenerator):
    async def generate_response(
        self,
        advice: Advice,
        request: AdviceRequestContext,
        categories: Sequence[str],
        preferred_kind: AdviceKind | None,
    ) -> str:
        selected_categories = ", ".join(
            categories) if categories else "general"
        preferred = preferred_kind.value if preferred_kind else "no specific kind"
        return (
            "Placeholder response: recommending "
            f"'{advice.name}' ({advice.kind.value}). "
            f"Categories matched: {selected_categories}. "
            f"Preferred kind: {preferred}."
        )


class TrivialAdviceIntentDetector(AdviceIntentDetector):
    async def detect_preferred_kind(self, user_message: str) -> AdviceKind | None:
        return None


class StaticAdviceCategoryClassifier(AdviceCategoryClassifier):
    def __init__(self, fallback_category: str = "general") -> None:
        self._fallback_category = fallback_category

    async def infer_categories(self, user_message: str) -> Sequence[str]:
        if user_message.strip():
            return (self._fallback_category,)
        return ()
