from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
from requests import HTTPError
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
    return (s or "").strip()


@retry(
    retry=retry_if_exception_type((requests.RequestException, HTTPError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.6, min=0.6, max=6),
    reraise=True,
)
def _http_get_json(url: str, timeout: int = 20) -> Any:
    headers = {
        "User-Agent": "slp-vehicle-defect-mvp/1.0",
        "Accept": "application/json",
    }

    resp = requests.get(url, timeout=timeout, headers=headers)

    # Treat 404 as empty dataset instead of hard failure
    if resp.status_code == 404:
        return {"results": []}

    # Retry on rate limits or server errors
    if resp.status_code in (429, 500, 502, 503, 504):
        resp.raise_for_status()

    resp.raise_for_status()

    try:
        return resp.json()
    except Exception as e:
        raise NHTSAError(f"Invalid JSON returned from NHTSA: {url}") from e


def get_json(
    url: str,
    cache: Optional[DiskCache] = None,
    ttl_seconds: int = 24 * 3600,
    timeout: int = 20,
) -> Any:
    if cache is not None:
        cached = cache.get(url, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    try:
        data = _http_get_json(url, timeout=timeout)
    except HTTPError as e:
        raise NHTSAError(f"NHTSA request failed after retries.\nURL: {url}\n{e}") from e
    except requests.RequestException as e:
        raise NHTSAError(f"NHTSA network error.\nURL: {url}\n{e}") from e

    if cache is not None:
        cache.set(url, data)

    return data


def decode_vin(vin: str, cache: Optional[DiskCache] = None) -> Dict[str, Any]:
    vin = (vin or "").strip().upper()
    if not vin:
        raise NHTSAError("VIN is required.")
    if len(vin) != 17:
        raise NHTSAError("VIN must be 17 characters.")

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{quote_plus(vin)}?format=json"
    payload = get_json(url, cache=cache, ttl_seconds=7 * 24 * 3600)

    results = payload.get("Results") or []
    if not results:
        raise NHTSAError("VIN decode returned no results.")

    row = results[0]
    err_code = str(row.get("ErrorCode", "")).strip()
    err_text = str(row.get("ErrorText", "")).strip()

    if err_code not in ("0", "0.0", "", "null", "None"):
        row["_vin_decode_warning"] = f"{err_code}: {err_text}"

    return row


def fetch_recalls_by_vehicle(
    make: str,
    model: str,
    year: int,
    cache: Optional[DiskCache] = None,
) -> List[Dict[str, Any]]:
    make = _clean_make_model(make)
    model = _clean_make_model(model)

    url = (
        "https://api.nhtsa.gov/recalls/recallsByVehicle"
        f"?make={quote_plus(make)}&model={quote_plus(model)}&modelYear={int(year)}"
    )

    payload = get_json(url, cache=cache, ttl_seconds=12 * 3600)
    return payload.get("results") or payload.get("Results") or []


def fetch_recalls_by_campaign(
    campaign_number: str,
    cache: Optional[DiskCache] = None,
) -> List[Dict[str, Any]]:
    campaign_number = (campaign_number or "").strip()
    if not campaign_number:
        return []

    url = (
        "https://api.nhtsa.gov/recalls/campaignNumber"
        f"?campaignNumber={quote_plus(campaign_number)}"
    )

    payload = get_json(url, cache=cache, ttl_seconds=24 * 3600)
    return payload.get("results") or payload.get("Results") or []


def fetch_complaints_by_vehicle(
    make: str,
    model: str,
    year: int,
    cache: Optional[DiskCache] = None,
) -> List[Dict[str, Any]]:
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

    issue_type = (issue_type or "").strip().lower()
    if issue_type not in {"complaints", "recalls", "investigations"}:
        raise NHTSAError(f"Unsupported issue_type: {issue_type}")

    nhtsa_id_str = str(nhtsa_id).strip()
    if not nhtsa_id_str:
        raise NHTSAError("nhtsa_id is required")

    url = (
        "https://api.nhtsa.gov/safetyIssues/byNhtsaId"
        f"?filter=issueType&filterValue={quote_plus(issue_type)}"
        f"&nhtsaId={quote_plus(nhtsa_id_str)}"
    )

    return get_json(url, cache=cache, ttl_seconds=7 * 24 * 3600)
