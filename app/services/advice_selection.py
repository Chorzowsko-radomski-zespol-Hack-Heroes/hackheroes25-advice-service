from __future__ import annotations

from dataclasses import dataclass
import asyncio
import logging
import random
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from app.models.advice import Advice, AdviceKind, AdviceRecommendation, AdviceRequestContext
from app.repositories.advice_repository import AdviceRepository
from app.repositories.category_repository import AdviceCategoryRepository
from app.integrations.openai import (
    OpenAISettings,
    create_async_openai_client,
    get_openai_settings,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from openai import AsyncOpenAI as OpenAIClient  # type: ignore[import]
else:
    OpenAIClient = Any  # type: ignore[misc]

try:  # pragma: no cover - runtime availability check
    from openai import AsyncOpenAI as _RuntimeAsyncOpenAI
except ImportError:
    _RuntimeAsyncOpenAI = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryMatch:
    name: str
    score: float
    rank: int


class AdviceCategoryClassifier(Protocol):
    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
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
        *,
        max_item_categories: int = 7,
    ) -> None:
        self._advice_repository = advice_repository
        self._category_repository = category_repository
        self._category_classifier = category_classifier
        self._intent_detector = intent_detector
        self._response_generator = response_generator
        self._max_item_categories = max_item_categories
        self._category_frequency_cache: dict[str, int] | None = None
        self._total_advice_count: int = 0
        self._frequency_lock = asyncio.Lock()

    async def recommend(self, request: AdviceRequestContext) -> AdviceRecommendation:
        category_matches = await self._infer_categories(request.user_message)
        preferred_kind = await self._intent_detector.detect_preferred_kind(
            request.user_message
        )

        advice = await self._select_advice(preferred_kind, category_matches)
        if advice is None:
            raise AdviceNotFoundError(
                "No advice found for the given criteria.")

        chat_response = await self._response_generator.generate_response(
            advice=advice,
            request=request,
            categories=tuple(match.name for match in category_matches),
            preferred_kind=preferred_kind,
        )
        return AdviceRecommendation(advice=advice, chat_response=chat_response)

    async def _infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        inferred_matches = await self._category_classifier.infer_categories(
            user_message
        )
        if not inferred_matches:
            logger.info(
                "Nie wykryto żadnych kategorii w wypowiedzi użytkownika.")
            return ()
        known_categories = await self._category_repository.get_all()
        variant_map = self._build_known_category_map(known_categories)
        unique_categories: list[CategoryMatch] = []
        seen = set()
        for match in inferred_matches:
            normalized = match.name.lower()
            matched_name = self._match_category_to_known(
                normalized, variant_map)
            if matched_name:
                canonical = matched_name
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    unique_categories.append(
                        CategoryMatch(
                            name=canonical,
                            score=match.score,
                            rank=match.rank,
                        )
                    )
                    logger.info(
                        "Wykryta kategoria '%s' | score=%.3f | pozycja=%d",
                        canonical,
                        match.score,
                        match.rank,
                    )
            else:
                logger.info(
                    "Kategoria '%s' nie pasuje do bazy (próbowano wariantów: %s)",
                    match.name,
                    self._build_category_variants(normalized),
                )
        if not unique_categories:
            logger.info(
                "Żadna wykryta kategoria nie istnieje w bazie kategorii.")
        return tuple(unique_categories)

    async def _select_advice(
        self,
        preferred_kind: AdviceKind | None,
        matched_categories: Sequence[CategoryMatch],
    ) -> Advice | None:
        candidates: Sequence[Advice] = ()
        category_names = tuple(match.name for match in matched_categories)

        if preferred_kind is not None:
            if matched_categories:
                candidates = await self._advice_repository.get_by_kind_and_containing_any_category(
                    preferred_kind,
                    category_names,
                )
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)
            if not candidates:
                candidates = await self._advice_repository.get_by_kind(preferred_kind)

        if not candidates:
            if matched_categories:
                all_advice = await self._advice_repository.get_all()
                candidates = tuple(
                    advice
                    for advice in all_advice
                    if self._contains_any_category(advice, category_names)
                )

        if not candidates:
            candidates = await self._advice_repository.get_all()

        await self._ensure_category_frequencies()
        selected = self._rank_candidates(
            candidates,
            matched_categories,
            self._category_frequency_cache or {},
            max(self._total_advice_count, len(candidates)),
        )
        if selected:
            logger.info("Pipeline selected advice name='%s'", selected.name)
        else:
            logger.warning(
                "Pipeline ranking returned None despite candidates.")
        return selected

    @staticmethod
    def _contains_any_category(advice: Advice, categories: Sequence[str]) -> bool:
        advice_categories = {category.lower()
                             for category in advice.categories}
        for category in categories:
            if category.lower() in advice_categories:
                return True
        return False

    def _rank_candidates(
        self,
        candidates: Sequence[Advice],
        matched_categories: Sequence[CategoryMatch],
        category_frequencies: Mapping[str, int],
        total_advice_count: int,
    ) -> Advice | None:
        if not candidates:
            return None

        if not matched_categories:
            return random.choice(tuple(candidates))

        match_map = {match.name.lower(): match for match in matched_categories}
        population: list[Advice] = []
        weights: list[float] = []

        for candidate in candidates:
            candidate_category_set = {
                category.lower() for category in candidate.categories
            }
            applicable_matches = [
                match_map[name]
                for name in candidate_category_set
                if name in match_map
            ]

            if not applicable_matches:
                population.append(candidate)
                weights.append(0.001)
                continue

            for match in applicable_matches:
                freq = category_frequencies.get(match.name.lower(), 0)
                if freq == 1 and match.rank == 1:
                    logger.info(
                        "Unique top category '%s' found only in advice '%s'; selecting deterministically.",
                        match.name,
                        candidate.name,
                    )
                    return candidate

            specificity = (self._max_item_categories + 1) / (
                len(candidate.categories) + 1
            )
            weight = 0.0
            for match in applicable_matches:
                freq = category_frequencies.get(match.name.lower(), 0) or 1
                rarity = ((total_advice_count + 1) / (freq + 1)) ** 2
                if freq == 1 and match.rank == 2:
                    rarity *= 12
                ranking_weight = (
                    len(matched_categories) - match.rank + 1
                ) / len(matched_categories)
                similarity = match.score ** 2
                incremental = similarity * ranking_weight * rarity * specificity
                weight += incremental

            weight *= random.uniform(0.9, 1.1)
            weights.append(max(weight, 0.02))
            population.append(candidate)

        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(tuple(candidates))

        selected = random.choices(
            population=population, weights=weights, k=1)[0]
        selected_index = population.index(selected)
        self._log_weights(population, weights,
                          selected_index, matched_categories)
        return selected

    @staticmethod
    @staticmethod
    def _match_category_to_known(
        candidate: str, variant_map: dict[str, str]
    ) -> str | None:
        for variant in AdviceSelectionPipeline._build_category_variants(candidate):
            mapped = variant_map.get(variant.lower())
            if mapped:
                return mapped
        return None

    @staticmethod
    def _build_category_variants(candidate: str) -> tuple[str, ...]:
        stripped = candidate.strip().lower()
        variants: list[str] = []
        ascii_variant = (
            unicodedata.normalize("NFKD", stripped)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        possible = [
            stripped,
            ascii_variant,
            stripped.replace(" ", "-"),
            ascii_variant.replace(" ", "-"),
            stripped.replace(" ", "_"),
            ascii_variant.replace(" ", "_"),
        ]
        for variant in possible:
            if variant and variant not in variants:
                variants.append(variant)
            sanitized = "".join(
                ch for ch in variant if ch.isalnum() or ch in {"-", "_", " "}
            ).strip()
            if sanitized and sanitized not in variants:
                variants.append(sanitized)
        return tuple(variants)

    @staticmethod
    def _build_known_category_map(categories: Sequence[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name in categories:
            for variant in AdviceSelectionPipeline._build_category_variants(name):
                key = variant.lower()
                mapping.setdefault(key, name)
        return mapping

    async def _ensure_category_frequencies(self) -> None:
        if self._category_frequency_cache is not None:
            return
        async with self._frequency_lock:
            if self._category_frequency_cache is not None:
                return
            all_advice = await self._advice_repository.get_all()
            counter: Counter[str] = Counter()
            for advice in all_advice:
                seen = set()
                for category in advice.categories:
                    normalized = category.lower()
                    if normalized not in seen:
                        counter[normalized] += 1
                        seen.add(normalized)
            self._category_frequency_cache = dict(counter)
            self._total_advice_count = len(all_advice)
            logger.info(
                "Zaktualizowano częstotliwości kategorii (liczba porad=%d).",
                self._total_advice_count,
            )

    @staticmethod
    def _log_weights(
        population: Sequence[Advice],
        weights: Sequence[float],
        selected_index: int,
        matched_categories: Sequence[CategoryMatch],
    ) -> None:
        lines = ["Podsumowanie wag dla kandydatów:"]
        for idx, (advice, weight) in enumerate(zip(population, weights)):
            marker = " <= WYBRANA" if idx == selected_index else ""
            lines.append(f"  - {advice.name}: waga={weight:.3f}{marker}")
        lines.append(
            "Kategorie podstawowe: "
            + ", ".join(
                f"{match.name} (rank={match.rank}, score={match.score:.3f})"
                for match in matched_categories
            )
        )
        logger.info("\n".join(lines))


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

    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        if user_message.strip():
            return (
                CategoryMatch(
                    name=self._fallback_category,
                    score=1.0,
                    rank=1,
                ),
            )
        return ()


@dataclass(frozen=True)
class EmbeddingCategoryDefinition:
    name: str
    description: str


class OpenAIEmbeddingCategoryClassifier(AdviceCategoryClassifier):
    def __init__(
        self,
        categories: Sequence[EmbeddingCategoryDefinition],
        *,
        settings: OpenAISettings | None = None,
        client: OpenAIClient | None = None,
        model: str | None = None,
        similarity_threshold: float = 0.25,
        max_categories: int | None = 6,
    ) -> None:
        if _RuntimeAsyncOpenAI is None:
            raise RuntimeError(
                "openai package not installed. Install it with `pip install openai`."
            )
        if not categories:
            raise ValueError(
                "OpenAIEmbeddingCategoryClassifier requires categories.")
        self._settings = settings or get_openai_settings()
        runtime_client: OpenAIClient = client or create_async_openai_client(
            self._settings
        )
        if not isinstance(runtime_client, _RuntimeAsyncOpenAI):
            raise TypeError(
                "client must be an instance of openai.AsyncOpenAI.")
        self._client = runtime_client
        self._model = model or self._settings.embeddings_model
        self._similarity_threshold = similarity_threshold
        self._max_categories = max_categories
        self._definitions = tuple(
            EmbeddingCategoryDefinition(
                name=definition.name.strip(),
                description=definition.description,
            )
            for definition in categories
        )
        self._category_embeddings: tuple[tuple[float, ...], ...] | None = None
        self._prepare_lock = asyncio.Lock()
        logger.info(
            "Initialized OpenAIEmbeddingCategoryClassifier with %d categories "
            "model=%s threshold=%.2f max_categories=%s",
            len(self._definitions),
            self._model,
            self._similarity_threshold,
            self._max_categories if self._max_categories is not None else "unlimited",
        )

    async def infer_categories(self, user_message: str) -> Sequence[CategoryMatch]:
        message = user_message.strip()
        if not message:
            return ()
        await self._ensure_category_embeddings()
        if not self._category_embeddings:
            return ()
        message_embedding = await self._embed_text(message)
        if not message_embedding:
            return ()

        scored_pairs: list[tuple[float, str]] = []
        for vector, definition in zip(self._category_embeddings, self._definitions):
            score = self._cosine_similarity(message_embedding, vector)
            if score >= self._similarity_threshold:
                scored_pairs.append((score, definition.name))

        scored_pairs.sort(reverse=True, key=lambda item: item[0])
        if scored_pairs:
            lines = [
                "Detected categories (similarity score):",
                *[
                    f"  - {slug}: {score:.3f}"
                    for score, slug in scored_pairs
                ],
            ]
            logger.info("\n".join(lines))
        else:
            logger.info(
                "No categories passed similarity threshold %.2f",
                self._similarity_threshold,
            )

        if self._max_categories is not None:
            scored_pairs = scored_pairs[: self._max_categories]

        matches = tuple(
            CategoryMatch(name=name, score=score, rank=index + 1)
            for index, (score, name) in enumerate(scored_pairs)
        )
        return matches

    async def _ensure_category_embeddings(self) -> None:
        if self._category_embeddings is not None:
            return
        async with self._prepare_lock:
            if self._category_embeddings is not None:
                return
            logger.info(
                "Embedding %d category descriptions via OpenAI model '%s'",
                len(self._definitions),
                self._model,
            )
            inputs = [definition.description for definition in self._definitions]
            embeddings = await self._embed_texts(inputs)
            self._category_embeddings = tuple(
                tuple(vector) for vector in embeddings)
            logger.info("Cached category embeddings.")

    async def _embed_text(self, text: str) -> tuple[float, ...]:
        embeddings = await self._embed_texts([text])
        if not embeddings:
            return ()
        return tuple(embeddings[0])

    async def _embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [tuple(item.embedding) for item in response.data]

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
