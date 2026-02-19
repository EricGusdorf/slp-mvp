from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .cache import DiskCache


class NHTSAError(RuntimeError):
    pass


@dataclass(frozen=True)
class VehicleKey:
    make: str
    model: str
    year: int


def _clean_make_model(s: str) -> str:
    s = (s or "").strip()
    # NHTSA endpoints appear to be case-insensitive for make/model, but be consistent.
    return s


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
)
def _http_get_json(url: str, timeout: int = 20) -> Any:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "slp-vehicle-defect-mvp/0.1"})
    resp.raise_for_status()
    return resp.json()


def get_json(url: str, cache: Optional[DiskCache] = None, ttl_seconds: int = 24 * 3600, timeout: int = 20) -> Any:
    if cache is not None:
        cached = cache.get(url, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached
    data = _http_get_json(url, timeout=timeout)
    if cache is not None:
        cache.set(url, data)
    return data


def decode_vin(vin: str, cache: Optional[DiskCache] = None) -> Dict[str, Any]:
    """
    Decode a VIN via vPIC. Returns dict with Make/Model/ModelYear etc.

    Uses:
      https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json
    """
    vin = (vin or "").strip().upper()
    if not vin:
        raise NHTSAError("VIN is required.")
    if len(vin) != 17:
        # vPIC can still return partial decode for shorter VINs, but this app expects full VIN.
        raise NHTSAError("VIN must be 17 characters.")

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{quote_plus(vin)}?format=json"
    payload = get_json(url, cache=cache, ttl_seconds=7 * 24 * 3600)
    results = payload.get("Results") or []
    if not results:
        raise NHTSAError("VIN decode returned no results.")
    row = results[0]
    # vPIC returns 'ErrorCode' and 'ErrorText' when invalid
    err_code = str(row.get("ErrorCode", "")).strip()
    err_text = str(row.get("ErrorText", "")).strip()
    if err_code not in ("0", "0.0", "", "null", "None"):
        # Not fatal in all cases, but signal it.
        row["_vin_decode_warning"] = f"{err_code}: {err_text}"
    return row


def fetch_recalls_by_vehicle(make: str, model: str, year: int, cache: Optional[DiskCache] = None) -> List[Dict[str, Any]]:
    make = _clean_make_model(make)
    model = _clean_make_model(model)
    url = (
        "https://api.nhtsa.gov/recalls/recallsByVehicle"
        f"?make={quote_plus(make)}&model={quote_plus(model)}&modelYear={int(year)}"
    )
    payload = get_json(url, cache=cache, ttl_seconds=12 * 3600)
    return payload.get("results") or payload.get("Results") or []


def fetch_recalls_by_campaign(campaign_number: str, cache: Optional[DiskCache] = None) -> List[Dict[str, Any]]:
    campaign_number = (campaign_number or "").strip()
    if not campaign_number:
        return []
    url = "https://api.nhtsa.gov/recalls/campaignNumber" + f"?campaignNumber={quote_plus(campaign_number)}"
    payload = get_json(url, cache=cache, ttl_seconds=24 * 3600)
    return payload.get("results") or payload.get("Results") or []


def fetch_complaints_by_vehicle(make: str, model: str, year: int, cache: Optional[DiskCache] = None) -> List[Dict[str, Any]]:
    make = _clean_make_model(make)
    model = _clean_make_model(model)
    url = (
        "https://api.nhtsa.gov/complaints/complaintsByVehicle"
        f"?make={quote_plus(make)}&model={quote_plus(model)}&modelYear={int(year)}"
    )
    payload = get_json(url, cache=cache, ttl_seconds=12 * 3600)
    return payload.get("results") or payload.get("Results") or []


def fetch_safety_issue_by_nhtsa_id(
    nhtsa_id: str | int,
    issue_type: str,
    cache: Optional[DiskCache] = None,
) -> Dict[str, Any]:
    """
    Fetch a "safety issue" record by NHTSA id, filtered by issueType.

    Observed usage:
      - complaints: https://api.nhtsa.gov/safetyIssues/byNhtsaId?filter=issueType&filterValue=complaints&nhtsaId=<ODI>
      - recalls:    ... filterValue=recalls&nhtsaId=<campaign>

    Returns raw JSON.
    """
    issue_type = (issue_type or "").strip().lower()
    if issue_type not in {"complaints", "recalls", "investigations"}:
        raise NHTSAError(f"Unsupported issue_type: {issue_type}")

    nhtsa_id_str = str(nhtsa_id).strip()
    if not nhtsa_id_str:
        raise NHTSAError("nhtsa_id is required")

    url = (
        "https://api.nhtsa.gov/safetyIssues/byNhtsaId"
        f"?filter=issueType&filterValue={quote_plus(issue_type)}&nhtsaId={quote_plus(nhtsa_id_str)}"
    )
    # These results are fairly stable; cache them longer.
    payload = get_json(url, cache=cache, ttl_seconds=7 * 24 * 3600)
    return payload
