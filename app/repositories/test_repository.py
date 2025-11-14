from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from supabase.client import AsyncClient  # type: ignore[import]
else:
    AsyncClient = Any  # type: ignore[misc]


class TestRepository:
    """
    Persists raw answers and derived trait scores for both tests.
    """

    def __init__(
        self,
        client: AsyncClient,
        psych_table: str = "psychology_test_responses",
        vocation_table: str = "vocational_test_responses",
        trait_table: str = "user_traits",
    ) -> None:
        self._client = client
        self._psych_table = psych_table
        self._vocation_table = vocation_table
        self._trait_table = trait_table

    async def save_psychology_test(
        self,
        user_id: str,
        closed_answers: Sequence[int],
        open_answers: Sequence[str],
        traits: Mapping[str, float],
    ) -> None:
        await self._client.table(self._psych_table).insert(
            {
                "user_id": user_id,
                "closed_answers": json.dumps(list(closed_answers)),
                "open_answers": json.dumps(list(open_answers)),
                "traits": json.dumps(traits),
            }
        ).execute()
        await self._save_traits(user_id, traits, "psychology")

    async def save_vocational_test(
        self,
        user_id: str,
        closed_answers: Sequence[int],
        open_answers: Sequence[str],
        traits: Mapping[str, float],
    ) -> None:
        await self._client.table(self._vocation_table).insert(
            {
                "user_id": user_id,
                "closed_answers": json.dumps(list(closed_answers)),
                "open_answers": json.dumps(list(open_answers)),
                "traits": json.dumps(traits),
            }
        ).execute()
        await self._save_traits(user_id, traits, "vocation")

    async def has_psychology_results(self, user_id: str) -> bool:
        response = (
            await self._client.table(self._psych_table)
            .select("id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    async def _save_traits(
        self,
        user_id: str,
        traits: Mapping[str, float],
        trait_type: str,
    ) -> None:
        await self._client.table(self._trait_table).insert(
            {
                "user_id": user_id,
                "trait_type": trait_type,
                "traits": json.dumps(traits),
            }
        ).execute()

    async def get_traits(self, user_id: str, trait_type: str) -> Mapping[str, float] | None:
        response = (
            await self._client.table(self._trait_table)
            .select("traits")
            .eq("user_id", user_id)
            .eq("trait_type", trait_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        records = response.data or []
        if not records:
            return None
        try:
            return json.loads(records[0]["traits"])
        except Exception:
            return None
