from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - optional dependency hint
    from supabase.client import AsyncClient  # type: ignore[import]
    from supabase.lib.client_options import ClientOptions  # type: ignore[import]
else:  # pragma: no cover - allow import without supabase installed
    AsyncClient = Any  # type: ignore
    ClientOptions = Any  # type: ignore

_CLIENT_INFO_HEADER: Final[dict[str, str]] = {
    "X-Client-Info": "hackheroes25-advice-service"
}


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    service_role_key: str

    @classmethod
    def from_env(cls) -> "SupabaseSettings":
        missing = []
        url = os.getenv("SUPABASE_URL")
        if not url:
            missing.append("SUPABASE_URL")

        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")

        if missing:
            env_list = ", ".join(missing)
            raise RuntimeError(
                f"Missing required Supabase environment variables: {env_list}"
            )

        return cls(url=url, service_role_key=service_role_key)


@lru_cache(maxsize=1)
def get_supabase_settings() -> SupabaseSettings:
    return SupabaseSettings.from_env()


def create_supabase_async_client(
    settings: SupabaseSettings | None = None,
) -> AsyncClient:
    settings = settings or get_supabase_settings()
    options: ClientOptions | None = None
    if ClientOptions is not Any:  # type: ignore[comparison-overlap]
        options = ClientOptions(
            headers=_CLIENT_INFO_HEADER,
            auto_refresh_token=False,
            persist_session=False,
        )
    return AsyncClient(
        supabase_url=settings.url,
        supabase_key=settings.service_role_key,
        options=options,
    )


