from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .utils import extract_state_abbr, split_components, safe_int


def recalls_to_df(recalls: List[Dict[str, Any]]) -> pd.DataFrame:
    if not recalls:
        return pd.DataFrame(columns=[
            "NHTSACampaignNumber", "Manufacturer", "Component", "Summary", "ReportReceivedDate",
            "PotentialNumberofUnitsAffected", "Remedy", "Notes"
        ])
    df = pd.DataFrame(recalls).copy()
    # Normalize common fields (API capitalization varies a bit by endpoint)
    # Keep original keys if present.
    return df


def complaints_to_df(complaints: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert /complaints/complaintsByVehicle results to a dataframe.
    """
    if not complaints:
        return pd.DataFrame(columns=[
            "odiNumber", "manufacturer", "crash", "fire", "numberOfInjuries", "numberOfDeaths",
            "dateOfIncident", "dateComplaintFiled", "vin", "components", "summary",
            "productYear", "productMake", "productModel"
        ])

    rows = []
    for c in complaints:
        row = dict(c)
        # Extract product info if present
        products = c.get("products") or []
        if products and isinstance(products, list):
            v = next((p for p in products if str(p.get("type", "")).lower() == "vehicle"), products[0])
            row["productYear"] = v.get("productYear")
            row["productMake"] = v.get("productMake")
            row["productModel"] = v.get("productModel")
        rows.append(row)

    df = pd.DataFrame(rows)
    # Normalize booleans / ints
    for col in ["crash", "fire"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in ["numberOfInjuries", "numberOfDeaths"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    # Parse dates (complaintsByVehicle uses MM/DD/YYYY)
    for col in ["dateOfIncident", "dateComplaintFiled"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def enrich_complaint_from_safety_issue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert /safetyIssues/byNhtsaId complaint response into a flat dict.

    Expected structure:
      {"results": [{"complaints": [ { ... } ] }]}
    """
    results = payload.get("results") or []
    if not results:
        return {}
    first = results[0] or {}
    complaints = first.get("complaints") or []
    if not complaints:
        return {}
    c = complaints[0] or {}
    out = {
        "odiNumber": c.get("nhtsaIdNumber") or c.get("odiId") or c.get("ODINumber"),
        "description": c.get("description"),
        "consumerLocation": c.get("consumerLocation"),
        "stateAbbreviation": extract_state_abbr(c.get("consumerLocation")),
        "dateOfIncident_iso": c.get("dateOfIncident"),
        "dateFiled_iso": c.get("dateFiled"),
        "crash": bool(c.get("crash", False)),
        "fire": bool(c.get("fire", False)),
        "numberOfInjuries": safe_int(c.get("numberOfInjuries", 0)),
        "numberOfDeaths": safe_int(c.get("numberOfDeaths", 0)),
    }

    # Components list is structured; keep both
    comps = c.get("components") or []
    if isinstance(comps, list):
        out["components_structured"] = comps
        out["components_names"] = [str(x.get("name", "")).upper() for x in comps if isinstance(x, dict)]
    else:
        out["components_structured"] = []
        out["components_names"] = []

    prods = c.get("associatedProducts") or []
    if isinstance(prods, list) and prods:
        v = next((p for p in prods if str(p.get("type", "")).lower() == "vehicle"), prods[0])
        out["productYear"] = v.get("productYear")
        out["productMake"] = v.get("productMake")
        out["productModel"] = v.get("productModel")
        out["manufacturer"] = v.get("manufacturer")
    return out


def component_frequency(complaints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe with columns [component, count, share]
    """
    if complaints_df is None or complaints_df.empty:
        return pd.DataFrame(columns=["component", "count", "share"])

    counts = {}
    for v in complaints_df.get("components", pd.Series(dtype=str)).fillna("").tolist():
        for comp in split_components(v):
            counts[comp] = counts.get(comp, 0) + 1

    total = sum(counts.values()) or 0
    rows = []
    for comp, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        rows.append({"component": comp, "count": cnt, "share": (cnt / total) if total else 0.0})
    return pd.DataFrame(rows)


def severity_summary(complaints_df: pd.DataFrame) -> Dict[str, int]:
    if complaints_df is None or complaints_df.empty:
        return {"crash": 0, "fire": 0, "injuries": 0, "deaths": 0}
    return {
        "crash": int(complaints_df.get("crash", False).sum()) if "crash" in complaints_df.columns else 0,
        "fire": int(complaints_df.get("fire", False).sum()) if "fire" in complaints_df.columns else 0,
        "injuries": int(complaints_df.get("numberOfInjuries", 0).sum()) if "numberOfInjuries" in complaints_df.columns else 0,
        "deaths": int(complaints_df.get("numberOfDeaths", 0).sum()) if "numberOfDeaths" in complaints_df.columns else 0,
    }


def complaints_time_series(complaints_df: pd.DataFrame, date_col: str = "dateComplaintFiled") -> pd.DataFrame:
    """
    Monthly counts for the chosen date column.
    """
    if complaints_df is None or complaints_df.empty or date_col not in complaints_df.columns:
        return pd.DataFrame(columns=["month", "count"])
    s = complaints_df[date_col].dropna()
    if s.empty:
        return pd.DataFrame(columns=["month", "count"])
    month = s.dt.to_period("M").dt.to_timestamp()
    ts = month.value_counts().sort_index()
    out = ts.rename_axis("month").reset_index(name="count")
    return out
