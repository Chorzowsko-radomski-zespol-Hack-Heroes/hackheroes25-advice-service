from __future__ import annotations

from typing import Literal, Protocol

DemandLevel = Literal["veryLow", "low", "medium", "high", "veryHigh"]
DemandType = Literal["current", "in5years"]


class JobDemandRepository(Protocol):
    """Repository for job demand data."""

    async def get_demand(
        self, job_name: str, demand_type: DemandType
    ) -> DemandLevel | None:
        """
        Pobiera poziom popytu dla danego zawodu.

        Args:
            job_name: Nazwa zawodu
            demand_type: Typ popytu - "current" (obecny) lub "in5years" (za 5 lat)

        Returns:
            Poziom popytu lub None jeśli zawód nie został znaleziony
        """
        raise NotImplementedError


class InMemoryJobDemandRepository(JobDemandRepository):
    """
    Implementacja repozytorium popytu zawodów używająca prostej mapy Python.
    Dane przechowywane jako słownik: {job_name: {"current": level, "in5years": level}}
    """

    def __init__(
        self, job_demands: dict[str, dict[DemandType, DemandLevel]] | None = None
    ) -> None:
        """
        Args:
            job_demands: Słownik z danymi popytu. Format:
                {
                    "nazwa_zawodu": {
                        "current": "high",
                        "in5years": "medium"
                    },
                    ...
                }
        """
        self._demands: dict[str, dict[DemandType,
                                      DemandLevel]] = job_demands or {}

    async def get_demand(
        self, job_name: str, demand_type: DemandType
    ) -> DemandLevel | None:
        job_data = self._demands.get(job_name)
        if not job_data:
            return None
        return job_data.get(demand_type)

    def add_job_demand(
        self,
        job_name: str,
        current_demand: DemandLevel | None = None,
        in5years_demand: DemandLevel | None = None,
    ) -> None:
        """
        Dodaje lub aktualizuje dane popytu dla zawodu.

        Args:
            job_name: Nazwa zawodu
            current_demand: Poziom popytu obecnego
            in5years_demand: Poziom popytu za 5 lat
        """
        if job_name not in self._demands:
            self._demands[job_name] = {}
        if current_demand is not None:
            self._demands[job_name]["current"] = current_demand
        if in5years_demand is not None:
            self._demands[job_name]["in5years"] = in5years_demand
