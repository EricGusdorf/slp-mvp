from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .cache import DiskCache
from .nhtsa import fetch_safety_issue_by_nhtsa_id
from .analytics import enrich_complaint_from_safety_issue


def enrich_complaints_df(
    complaints_df: pd.DataFrame,
    cache: Optional[DiskCache],
    max_records: int = 150,
    max_workers: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich a complaints dataframe with additional fields from /safetyIssues/byNhtsaId:
      - consumerLocation, stateAbbreviation
      - full description
      - structured components
      - ISO dates

    Returns (enriched_df, stats)
    """
    if complaints_df is None or complaints_df.empty or "odiNumber" not in complaints_df.columns:
        return complaints_df, {"requested": 0, "enriched": 0, "failed": 0}

    df = complaints_df.copy()
    odi_numbers = [x for x in df["odiNumber"].dropna().astype(int).tolist()]
    odi_numbers = odi_numbers[: max(0, int(max_records))]

    results = {}
    failed = 0

    def _fetch_one(odi: int) -> Dict[str, Any]:
        payload = fetch_safety_issue_by_nhtsa_id(odi, issue_type="complaints", cache=cache)
        return enrich_complaint_from_safety_issue(payload)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, odi): odi for odi in odi_numbers}
        for fut in as_completed(futs):
            odi = futs[fut]
            try:
                row = fut.result()
                if row:
                    results[int(odi)] = row
                else:
                    failed += 1
            except Exception:
                failed += 1

    # Merge back
    if results:
        enrich_rows = pd.DataFrame(list(results.values()))
        # Avoid duplicate columns; suffix enriched ones
        df = df.merge(enrich_rows, how="left", on="odiNumber", suffixes=("", "_enriched"))

        # Parse ISO dates if present
        if "dateFiled_iso" in df.columns:
            df["dateFiled_iso"] = pd.to_datetime(df["dateFiled_iso"], errors="coerce")
        if "dateOfIncident_iso" in df.columns:
            df["dateOfIncident_iso"] = pd.to_datetime(df["dateOfIncident_iso"], errors="coerce")

        # If missing consumerLocation/stateAbbreviation, keep as is
        if "stateAbbreviation" in df.columns:
            pass

    return df, {"requested": len(odi_numbers), "enriched": len(results), "failed": failed}
