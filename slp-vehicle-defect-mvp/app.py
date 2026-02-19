import os
import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

from slp_mvp.cache import DiskCache
from slp_mvp.nhtsa import (
    decode_vin,
    fetch_complaints_by_vehicle,
    fetch_recalls_by_vehicle,
    NHTSAError,
)
from slp_mvp.analytics import (
    complaints_to_df,
    recalls_to_df,
    component_frequency,
    severity_summary,
    complaints_time_series,
)
from slp_mvp.enrich import enrich_complaints_df
from slp_mvp.text_search import build_index, search as search_index


@st.cache_data(ttl=7 * 24 * 3600)
def vp_get_all_makes() -> list[str]:
    try:
        url = "https://vpic.nhtsa.dot.gov/api/vehicles/getallmakes?format=json"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        makes = [row.get("Make_Name", "").strip() for row in (data.get("Results") or [])]
        makes = sorted({m for m in makes if m})

        common = [
            "Acura",
            "Alfa Romeo",
            "Audi",
            "BMW",
            "Buick",
            "Cadillac",
            "Chevrolet",
            "Chrysler",
            "Dodge",
            "Fiat",
            "Ford",
            "Genesis",
            "GMC",
            "Honda",
            "Hyundai",
            "Infiniti",
            "Jaguar",
            "Jeep",
            "Kia",
            "Land Rover",
            "Lexus",
            "Lincoln",
            "Mazda",
            "Mercedes-Benz",
            "Mini",
            "Mitsubishi",
            "Nissan",
            "Polestar",
            "Porsche",
            "Ram",
            "Rivian",
            "Subaru",
            "Tesla",
            "Toyota",
            "Volkswagen",
            "Volvo",
        ]

        makes_by_lower = {m.lower(): m for m in makes}
        common_present = [makes_by_lower[c.lower()] for c in common if c.lower() in makes_by_lower]

        common_set = {m.lower() for m in common_present}
        rest = [m for m in makes if m.lower() not in common_set]

        return common_present + rest
    except Exception:
        return []


@st.cache_data(ttl=7 * 24 * 3600)
def vp_get_models_for_make_year(make: str, year: int) -> list[str]:
    try:
        make_q = quote_plus(make.strip())
        url = (
            "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/"
            f"make/{make_q}/modelyear/{int(year)}?format=json"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        models = [row.get("Model_Name", "").strip() for row in (data.get("Results") or [])]
        return sorted({m for m in models if m})
    except Exception:
        return []


@st.cache_data(ttl=7 * 24 * 3600)
def vp_get_models_for_make(make: str) -> list[str]:
    """
    vPIC's make+year model list can be incomplete for some makes/years.
    This broader endpoint helps users find valid model strings (e.g., hybrids/EVs).
    """
    try:
        make_q = quote_plus(make.strip())
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/{make_q}?format=json"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        models = [row.get("Model_Name", "").strip() for row in (data.get("Results") or [])]
        return sorted({m for m in models if m})
    except Exception:
        return []


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _candidate_models(make: str, model: str, year: int) -> list[str]:
    """
    NHTSA's `*ByVehicle` endpoints can be picky about exact model strings.
    If the chosen model returns no data, try a few close vPIC variants for the same make/year.
    """
    model = (model or "").strip()
    make = (make or "").strip()
    if not make or not model or not year:
        return []

    models = vp_get_models_for_make_year(make, int(year))
    # Some variants (e.g., hybrids) may not appear in the make+year list.
    models_all = vp_get_models_for_make(make)
    if models_all:
        models = sorted({*models, *models_all})
    if not models:
        return []

    norm_model = _normalize(model)
    if not norm_model:
        return []

    has_digits = bool(re.search(r"\d", model))
    first_digit_match = re.search(r"\d", model)
    first_digit = first_digit_match.group(0) if first_digit_match else ""
    digit_run_match = re.search(r"\d+", model)
    digit_run = digit_run_match.group(0) if digit_run_match else ""

    tok_model = (model.lower().split() or [""])[0]

    scored: list[tuple[int, str]] = []
    for m in models:
        if m.lower() == model.lower():
            continue

        nm = _normalize(m)
        score = 0

        # General similarity (helps cases like "Accord" vs "Accord Hybrid")
        if nm == norm_model:
            score += 10
        if norm_model in nm:
            score += 6
        if nm in norm_model:
            score += 2

        tok_m = (m.lower().split() or [""])[0]
        if tok_model and tok_m and tok_model == tok_m:
            score += 2

        # Digit-heavy trims benefit from extra heuristics (e.g., "535i" → "5-Series")
        if has_digits:
            if "series" in nm:
                score += 5
            if first_digit and nm.startswith(first_digit):
                score += 5
            if digit_run and digit_run in nm:
                score += 3

        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda x: (-x[0], x[1]))

    out: list[str] = []
    seen = set()
    for _, m in scored:
        ml = m.lower()
        if ml not in seen:
            out.append(m)
            seen.add(ml)
        if len(out) >= 6:
            break

    return out


st.set_page_config(
    page_title="Strategic Legal Practices  |  Vehicle Defect Assessment Tool",
    layout="wide",
)

# Force horizontal scrollbar always visible on dataframes
st.markdown(
    """
    <style>
    /* ---- Streamlit DataFrame (Glide Data Grid) scrollbar always visible ---- */
    div[data-testid="stDataFrame"] .gdg-scrollbar,
    div[data-testid="stDataFrame"] .gdg-scrollbar-horizontal,
    div[data-testid="stDataFrame"] .gdg-scrollbar-vertical,
    div[data-testid*="stDataFrame"] [class*="gdg-scrollbar"] {
        opacity: 1 !important;
        transition: none !important;
    }

    div[data-testid="stDataFrame"] .gdg-scrollbar-horizontal,
    div[data-testid*="stDataFrame"] [class*="gdg-scrollbar-horizontal"] {
        height: 14px !important;
    }

    div[data-testid="stDataFrame"] div[role="grid"],
    div[data-testid*="stDataFrame"] div[role="grid"] {
        overflow-x: auto !important;
    }

    /* Fallback (native scrollbars, if used by Streamlit version/browser) */
    div[data-testid*="stDataFrame"] *::-webkit-scrollbar {
        height: 14px;
        width: 14px;
    }
    div[data-testid*="stDataFrame"] *::-webkit-scrollbar-thumb {
        background: rgba(107, 114, 128, 0.55); /* gray-500-ish */
        border-radius: 999px;
    }
    div[data-testid*="stDataFrame"] *::-webkit-scrollbar-track {
        background: rgba(107, 114, 128, 0.15);
    }

    /* ---- Hide Streamlit header "link" (permalink) icons ---- */
    .stHeadingAnchor,
    .stMarkdownHeadingAnchor,
    a.stMarkdownHeaderAnchor,
    a.header-anchor,
    h1 a[href^="#"],
    h2 a[href^="#"],
    h3 a[href^="#"],
    h4 a[href^="#"],
    h5 a[href^="#"],
    h6 a[href^="#"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }

    /* ---- Simple section headings (no anchor icons) ---- */
    .slp-section-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }

    /* ---- Recalls table (always-visible horizontal scrollbar) ---- */
    .slp-recalls-scroll {
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-gutter: stable;
        padding-bottom: 6px; /* keeps bar from feeling clipped */
    }
    .slp-recalls-scroll table {
        border-collapse: collapse;
        width: max-content;
        min-width: 100%;
    }
    .slp-recalls-scroll th,
    .slp-recalls-scroll td {
        padding: 0.35rem 0.55rem;
        border-bottom: 1px solid rgba(107, 114, 128, 0.25);
        white-space: nowrap;
        vertical-align: top;
        font-size: 0.9rem;
    }
    .slp-recalls-scroll th {
        font-weight: 600;
        background: rgba(107, 114, 128, 0.08);
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .slp-recalls-scroll *::-webkit-scrollbar {
        height: 14px;
    }
    .slp-recalls-scroll *::-webkit-scrollbar-thumb {
        background: rgba(107, 114, 128, 0.55);
        border-radius: 999px;
    }
    .slp-recalls-scroll *::-webkit-scrollbar-track {
        background: rgba(107, 114, 128, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    ">
        Strategic Legal Practices  |  Vehicle Defect Assessment Tool
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CACHE_DIR = os.environ.get("SLP_CACHE_DIR", ".cache")
cache = DiskCache(DEFAULT_CACHE_DIR)


# --- Sidebar: vehicle selection ---
with st.sidebar:
    st.header("Vehicle input")
    input_mode = st.radio("Lookup by", ["VIN", "Make / Model / Year"], horizontal=False)

    analyze_clicked = False

    draft_vin = ""
    draft_make = ""
    draft_model = ""
    draft_year: Optional[int] = None

    if input_mode == "VIN":
        draft_vin = st.text_input(
            "VIN (17 chars)",
            value=st.session_state.get("draft_vin", ""),
            placeholder="e.g., 1HGCV1F56MA123456",
            key="draft_vin",
        )
        st.divider()
        analyze_clicked = st.button("Analyze vehicle", type="primary")
    else:
        draft_year = st.number_input(
            "Model year",
            min_value=1950,
            max_value=datetime.now().year + 1,
            value=st.session_state.get("draft_year", 2021),
            step=1,
            key="draft_year",
        )

        makes = vp_get_all_makes()

        draft_make = st.selectbox(
            "Make",
            options=[""] + makes,
            index=0,
            key="draft_make",
        )

        # Reset model only when make changes (not when year changes)
        prev_make = st.session_state.get("_prev_draft_make", "")
        if draft_make != prev_make:
            st.session_state["draft_model"] = ""
            st.session_state["draft_model_manual"] = ""
            st.session_state["_prev_draft_make"] = draft_make

        include_all_models = bool(st.session_state.get("draft_model_include_all", False))

        models: list[str] = []
        if draft_make:
            models = vp_get_models_for_make_year(draft_make, int(draft_year))
            if include_all_models:
                models = sorted({*models, *vp_get_models_for_make(draft_make)})

        # Keep current model if still valid for this make+year; otherwise clear
        current_model = st.session_state.get("draft_model", "")
        if current_model and current_model not in models:
            st.session_state["draft_model"] = ""

        options = [""] + models
        selected_model = st.session_state.get("draft_model", "")
        model_index = options.index(selected_model) if selected_model in options else 0

        manual_text_existing = (st.session_state.get("draft_model_manual") or "").strip()
        if "draft_model_manual_enabled" not in st.session_state:
            st.session_state["draft_model_manual_enabled"] = bool(manual_text_existing)

        draft_model = st.selectbox(
            "Model",
            options=options,
            index=model_index,
            key="draft_model",
            disabled=(not draft_make) or bool(manual_text_existing),
        )

        st.checkbox(
            "Show more models (Hybrid/EV/variants)",
            value=st.session_state.get("draft_model_include_all", False),
            key="draft_model_include_all",
            disabled=(not draft_make),
            help="Adds extra model names that sometimes don't show up for the selected year.",
        )

        manual_enabled = st.checkbox(
            "Model not listed? Enter model manually",
            value=st.session_state.get("draft_model_manual_enabled", False),
            key="draft_model_manual_enabled",
            disabled=(not draft_make),
        )
        if manual_enabled or bool(manual_text_existing):
            st.text_input(
                "Model (manual override)",
                value=st.session_state.get("draft_model_manual", ""),
                key="draft_model_manual",
                placeholder="e.g., Accord Hybrid",
                help="If you type something here, it will be used when you click Analyze vehicle.",
            )
            if (st.session_state.get("draft_model_manual") or "").strip():
                st.caption("Using manual model text for analysis. Clear it to re-enable the dropdown.")

        st.divider()
        analyze_clicked = st.button("Analyze vehicle", type="primary")

# Enrichment always on (no UI toggle)
enrich = True

ENRICH_LIMIT = int(os.environ.get("SLP_ENRICH_LIMIT", "120"))
ENRICH_WORKERS = int(os.environ.get("SLP_ENRICH_WORKERS", "6"))


def _set_analysis_error(message: str, details: Optional[str] = None) -> None:
    st.session_state["analysis_error"] = message
    st.session_state["analysis_error_details"] = details

    # Clear data so we never fall back to stale results from prior runs.
    st.session_state["analysis_recalls_df"] = pd.DataFrame()
    st.session_state["analysis_complaints_df"] = pd.DataFrame()
    st.session_state["analysis_enrich_stats"] = {"requested": 0, "enriched": 0, "failed": 0}
    st.session_state["analysis_raw_recalls"] = []
    st.session_state["analysis_raw_complaints"] = []


def _clear_analysis_error() -> None:
    st.session_state["analysis_error"] = None
    st.session_state["analysis_error_details"] = None


def _vehicle_from_vin(vin: str):
    decoded = decode_vin(vin, cache=cache)
    make_ = (decoded.get("Make") or "").strip()
    model_ = (decoded.get("Model") or "").strip()
    year_ = (decoded.get("ModelYear") or "").strip()
    warn = decoded.get("_vin_decode_warning")
    return make_, model_, int(year_) if str(year_).isdigit() else None, decoded, warn


def _best_text_column(df: pd.DataFrame) -> str:
    for col in ["description", "summary"]:
        if col in df.columns:
            return col
    return "summary"


# --- Run analysis ---
if analyze_clicked:
    try:
        _clear_analysis_error()

        if input_mode == "VIN":
            if not draft_vin:
                _set_analysis_error("Enter a VIN.")
                st.error(st.session_state["analysis_error"])
                st.stop()

            make_, model_, year_, decoded, warn = _vehicle_from_vin(draft_vin)
            if warn:
                st.info(f"VIN decode warning: {warn}")
            if not make_ or not model_ or not year_:
                st.session_state["analysis_vehicle"] = {
                    "make": make_,
                    "model": model_,
                    "year": year_,
                    "vin": draft_vin,
                    "decoded": decoded,
                }
                _set_analysis_error("Could not decode make/model/year from VIN. Try Make/Model/Year input.")
                st.error(st.session_state["analysis_error"])
                st.stop()

            st.session_state["analysis_vehicle"] = {
                "make": make_,
                "model": model_,
                "year": year_,
                "vin": draft_vin,
                "decoded": decoded,
            }

        else:
            model_for_lookup = draft_model
            manual = (st.session_state.get("draft_model_manual") or "").strip()
            if manual:
                model_for_lookup = manual

            if not draft_make or not model_for_lookup or not draft_year:
                st.session_state["analysis_vehicle"] = {
                    "make": draft_make,
                    "model": model_for_lookup,
                    "year": int(draft_year) if draft_year else None,
                    "vin": None,
                    "decoded": None,
                }
                _set_analysis_error("Enter make, model, and year.")
                st.error(st.session_state["analysis_error"])
                st.stop()

            st.session_state["analysis_vehicle"] = {
                "make": draft_make,
                "model": model_for_lookup,
                "year": int(draft_year),
                "vin": None,
                "decoded": None,
            }

        v = st.session_state["analysis_vehicle"]

        # Fetch with fallback model tries
        with st.spinner("Fetching NHTSA recalls + complaints..."):
            recalls: list[dict] = []
            complaints: list[dict] = []
            recalls_err: Optional[str] = None
            complaints_err: Optional[str] = None

            tried_models = [v["model"]]
            for m in _candidate_models(v["make"], v["model"], v["year"]):
                if m not in tried_models:
                    tried_models.append(m)

            used_model = v["model"]

            for m in tried_models:
                used_model = m
                recalls_err = None
                complaints_err = None

                try:
                    recalls = fetch_recalls_by_vehicle(v["make"], m, v["year"], cache=cache)
                except NHTSAError as e:
                    recalls_err = str(e)
                    recalls = []

                try:
                    complaints = fetch_complaints_by_vehicle(v["make"], m, v["year"], cache=cache)
                except NHTSAError as e:
                    complaints_err = str(e)
                    complaints = []

                if recalls or complaints:
                    break

        # Both endpoints failed => service issue
        if recalls_err and complaints_err:
            _set_analysis_error(
                "NHTSA services are currently unavailable for this lookup. "
                "Please confirm the vehicle exists and try again.",
                details=f"recalls error:\n{recalls_err}\n\ncomplaints error:\n{complaints_err}",
            )
            st.error(st.session_state["analysis_error"])
            st.stop()

        # If fallback found results, update model and inform user (blue)
        if (used_model != v["model"]) and (recalls or complaints):
            original_model = v["model"]
            st.session_state["analysis_vehicle"]["model"] = used_model
            st.info(f"No results for model '{original_model}'. Showing results for closest match '{used_model}'.")
            v = st.session_state["analysis_vehicle"]

        # No errors + still no data => treat as invalid/no data for this vehicle
        if (recalls_err is None) and (complaints_err is None) and (not recalls) and (not complaints):
            _set_analysis_error(
                f"No NHTSA data found for {v['year']} {v['make']} {v['model']}. "
                "Verify the make, model, and year."
            )
            st.error(st.session_state["analysis_error"])
            st.stop()

        # Partial failures (keep blue)
        if recalls_err and not complaints_err:
            st.info("No recalls found; showing complaints only.")
        if complaints_err and not recalls_err:
            st.info("No complaints found; showing recalls only.")

        st.session_state["analysis_raw_recalls"] = recalls
        st.session_state["analysis_raw_complaints"] = complaints

        recalls_df = recalls_to_df(recalls)
        complaints_df = complaints_to_df(complaints)

        enrich_stats = {"requested": 0, "enriched": 0, "failed": 0}
        if not complaints_df.empty:
            with st.spinner("Enriching complaints (location + full text)..."):
                complaints_df, enrich_stats = enrich_complaints_df(
                    complaints_df,
                    cache=cache,
                    max_records=int(ENRICH_LIMIT),
                    max_workers=int(ENRICH_WORKERS),
                )

        st.session_state["analysis_recalls_df"] = recalls_df
        st.session_state["analysis_complaints_df"] = complaints_df
        st.session_state["analysis_enrich_stats"] = enrich_stats

    except NHTSAError as e:
        _set_analysis_error(str(e))
        st.error(st.session_state["analysis_error"])
        st.stop()
    except Exception:
        # Avoid showing redacted stack traces to end users
        _set_analysis_error("Unexpected error. Please try again.")
        st.error(st.session_state["analysis_error"])
        st.stop()


# --- Display results if available ---
# Back-compat: if older keys exist (from prior session) but new ones don't, reuse them.
if ("analysis_vehicle" not in st.session_state) and ("vehicle" in st.session_state):
    st.session_state["analysis_vehicle"] = st.session_state["vehicle"]
    st.session_state["analysis_recalls_df"] = st.session_state.get("recalls_df", pd.DataFrame())
    st.session_state["analysis_complaints_df"] = st.session_state.get("complaints_df", pd.DataFrame())
    st.session_state["analysis_enrich_stats"] = st.session_state.get(
        "enrich_stats", {"requested": 0, "enriched": 0, "failed": 0}
    )
    st.session_state["analysis_raw_recalls"] = st.session_state.get("raw_recalls", [])
    st.session_state["analysis_raw_complaints"] = st.session_state.get("raw_complaints", [])

if "analysis_vehicle" in st.session_state:
    v = st.session_state["analysis_vehicle"]
    recalls_df = st.session_state.get("analysis_recalls_df", pd.DataFrame())
    complaints_df = st.session_state.get("analysis_complaints_df", pd.DataFrame())
    enrich_stats = st.session_state.get(
        "analysis_enrich_stats", {"requested": 0, "enriched": 0, "failed": 0}
    )

    analysis_error = st.session_state.get("analysis_error")
    analysis_error_details = st.session_state.get("analysis_error_details")
    if analysis_error:
        st.error(analysis_error)
        if analysis_error_details:
            with st.expander("Details"):
                st.code(analysis_error_details)
        st.stop()

    st.markdown(
        f"""
        <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.5rem;">
            {v['year']} 
            <span style="color:#6b7280; font-weight:500;">
                {v['make']} {v['model']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if v.get("vin"):
        st.caption(f"VIN: `{v['vin']}`")

    sev = severity_summary(complaints_df)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Complaints", f"{len(complaints_df):,}")
    k2.metric("Recalls", f"{len(recalls_df):,}")
    k3.metric("Crashes", f"{sev['crash']:,}")
    k4.metric("Fires", f"{sev['fire']:,}")
    k5.metric("Injuries", f"{sev['injuries']:,}")
    k6.metric("Deaths", f"{sev['deaths']:,}")

    tabs = st.tabs(["Summary", "Search", "Map", "Trends"])

    # --- Summary ---
    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            st.markdown('<div class="slp-section-title">Defect patterns</div>', unsafe_allow_html=True)
            comp_df = component_frequency(complaints_df)
            if comp_df.empty:
                st.info("No complaint component labels returned for this vehicle.")
            else:
                top_n = comp_df.head(10)

                fig = px.bar(
                    top_n,
                    x="count",
                    y="component",
                    orientation="h",
                    title="Top complaint components",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

                display_df = top_n[["component", "count", "share"]].copy()
                display_df = display_df.rename(
                    columns={
                        "component": "Component",
                        "count": "Complaint Count",
                        "share": "Share of Total Complaints (%)",
                    }
                )
                display_df["Share of Total Complaints (%)"] = (
                    display_df["Share of Total Complaints (%)"] * 100
                ).round(1)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown('<div class="slp-section-title">Recalls</div>', unsafe_allow_html=True)
            if recalls_df is None or recalls_df.empty:
                st.info("No recalls returned by NHTSA.")
            else:
                cols = [
                    c
                    for c in ["NHTSACampaignNumber", "ReportReceivedDate", "Component", "Summary"]
                    if c in recalls_df.columns
                ]

                recalls_display = recalls_df[cols].copy()

                if "ReportReceivedDate" in recalls_display.columns:
                    recalls_display["ReportReceivedDate"] = (
                        pd.to_datetime(recalls_display["ReportReceivedDate"], errors="coerce").dt.strftime("%m/%d/%Y")
                    )

                recalls_html = recalls_display.head(50).to_html(index=False, escape=True)
                components.html(
                    f"""
                    <div id="slp-recalls-container" style="width: 100%;">
                      <style>
                        :root {{
                          --track: rgba(147, 197, 253, 0.26);       /* even lighter blue */
                          --thumb: rgba(147, 197, 253, 0.72);
                          --thumb-hover: rgba(96, 165, 250, 0.92);
                          --border: rgba(147, 197, 253, 0.32);
                          --header: rgba(147, 197, 253, 0.16);
                          --row-alt: rgba(147, 197, 253, 0.08);
                          --text: rgba(30, 64, 175, 0.95);          /* keep readable */
                          --text-muted: rgba(37, 99, 235, 0.92);
                        }}

                        #slp-recalls-scroll {{
                          overflow-x: scroll;   /* force scroll container */
                          overflow-y: hidden;
                          width: 100%;
                          scrollbar-width: none; /* hide native scrollbar (Firefox) */
                        }}
                        #slp-recalls-scroll::-webkit-scrollbar {{
                          height: 0px;          /* hide native scrollbar (WebKit) */
                        }}

                        #slp-recalls-scroll table {{
                          border-collapse: collapse;
                          width: max-content;
                          min-width: 100%;
                          font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
                          font-size: 0.9rem;
                        }}
                        #slp-recalls-scroll th,
                        #slp-recalls-scroll td {{
                          padding: 0.35rem 0.55rem;
                          border-bottom: 1px solid var(--border);
                          white-space: nowrap;
                          vertical-align: top;
                          text-align: left;
                          color: var(--text);
                        }}
                        #slp-recalls-scroll th {{
                          font-weight: 600;
                          background: var(--header);
                          position: sticky;
                          top: 0;
                          z-index: 1;
                          color: var(--text-muted);
                        }}
                        #slp-recalls-scroll tbody tr:nth-child(even) td {{
                          background: var(--row-alt);
                        }}

                        /* Always-visible custom scrollbar */
                        #slp-recalls-bar {{
                          height: 14px;
                          background: var(--track);
                          border-radius: 999px;
                          margin-top: 6px;
                          position: relative;
                          user-select: none;
                          touch-action: none;
                        }}
                        #slp-recalls-thumb {{
                          height: 14px;
                          background: var(--thumb);
                          border-radius: 999px;
                          width: 40px;
                          transform: translateX(0px);
                          position: absolute;
                          left: 0;
                          top: 0;
                          cursor: pointer;
                        }}
                        #slp-recalls-thumb:hover {{
                          background: var(--thumb-hover);
                        }}
                      </style>

                      <div id="slp-recalls-scroll">{recalls_html}</div>
                      <div id="slp-recalls-bar" aria-label="Horizontal scroll bar">
                        <div id="slp-recalls-thumb" aria-label="Scroll thumb"></div>
                      </div>
                    </div>

                    <script>
                      const scrollEl = document.getElementById("slp-recalls-scroll");
                      const barEl = document.getElementById("slp-recalls-bar");
                      const thumbEl = document.getElementById("slp-recalls-thumb");

                      function clamp(v, min, max) {{
                        return Math.max(min, Math.min(max, v));
                      }}

                      function updateThumb() {{
                        const scrollWidth = scrollEl.scrollWidth;
                        const clientWidth = scrollEl.clientWidth;
                        const maxScroll = Math.max(0, scrollWidth - clientWidth);
                        const barWidth = barEl.clientWidth;

                        if (maxScroll <= 0) {{
                          barEl.style.display = "none";
                          return;
                        }}
                        barEl.style.display = "block";

                        const ratio = clientWidth / scrollWidth;
                        const thumbWidth = clamp(Math.round(barWidth * ratio), 28, barWidth);
                        thumbEl.style.width = thumbWidth + "px";

                        const maxThumbX = Math.max(0, barWidth - thumbWidth);
                        const x = maxScroll ? (scrollEl.scrollLeft / maxScroll) * maxThumbX : 0;
                        thumbEl.style.transform = `translateX(${{x}}px)`;
                      }}

                      scrollEl.addEventListener("scroll", updateThumb, {{ passive: true }});
                      window.addEventListener("resize", updateThumb);

                      let dragging = false;
                      let dragOffset = 0;

                      thumbEl.addEventListener("pointerdown", (e) => {{
                        dragging = true;
                        thumbEl.setPointerCapture(e.pointerId);
                        dragOffset = e.clientX - thumbEl.getBoundingClientRect().left;
                      }});

                      thumbEl.addEventListener("pointermove", (e) => {{
                        if (!dragging) return;
                        const barRect = barEl.getBoundingClientRect();
                        const thumbRect = thumbEl.getBoundingClientRect();
                        const barWidth = barRect.width;
                        const thumbWidth = thumbRect.width;
                        const maxThumbX = Math.max(0, barWidth - thumbWidth);
                        let x = e.clientX - barRect.left - dragOffset;
                        x = clamp(x, 0, maxThumbX);

                        const maxScroll = Math.max(0, scrollEl.scrollWidth - scrollEl.clientWidth);
                        scrollEl.scrollLeft = maxThumbX ? (x / maxThumbX) * maxScroll : 0;
                      }});

                      thumbEl.addEventListener("pointerup", () => {{
                        dragging = false;
                      }});

                      barEl.addEventListener("pointerdown", (e) => {{
                        if (e.target === thumbEl) return;
                        const barRect = barEl.getBoundingClientRect();
                        const barWidth = barRect.width;
                        const thumbWidth = thumbEl.getBoundingClientRect().width;
                        const maxThumbX = Math.max(0, barWidth - thumbWidth);
                        let x = e.clientX - barRect.left - thumbWidth / 2;
                        x = clamp(x, 0, maxThumbX);

                        const maxScroll = Math.max(0, scrollEl.scrollWidth - scrollEl.clientWidth);
                        scrollEl.scrollLeft = maxThumbX ? (x / maxThumbX) * maxScroll : 0;
                      }});

                      // Initial layout
                      updateThumb();
                      // A second update after layout settles (fonts/table rendering)
                      setTimeout(updateThumb, 50);
                    </script>
                    """,
                    height=420,
                    scrolling=False,
                )

        with st.expander("View complaints (all)"):
            if complaints_df is None or complaints_df.empty:
                st.info("No complaints returned by NHTSA.")
            else:
                cols = [
                    c
                    for c in [
                        "odiNumber",
                        "dateComplaintFiled",
                        "components",
                        "crash",
                        "fire",
                        "numberOfInjuries",
                        "numberOfDeaths",
                        "summary",
                    ]
                    if c in complaints_df.columns
                ]
                st.dataframe(complaints_df[cols], use_container_width=True, hide_index=True)

    # --- Search ---
    with tabs[1]:
        st.write("Search within this vehicle's NHTSA complaints by symptom text.")
        text_col = _best_text_column(complaints_df)
        if complaints_df.empty:
            st.info("No complaints for this vehicle.")
        else:
            query = st.text_input("Symptom query", value="", placeholder="e.g., transmission slipping")
            top_k = 10

            texts = complaints_df[text_col].fillna("").astype(str).tolist()
            idx = build_index(texts)
            matches = search_index(query, idx, top_k=int(top_k)) if query else []

            if query and not matches:
                st.info("No matches found.")
            elif query:
                rows = []
                for i, score_ in matches:
                    row = complaints_df.iloc[i].to_dict()
                    row["_matchScore"] = round(score_, 3)
                    rows.append(row)
                out = pd.DataFrame(rows)
                keep = [
                    c
                    for c in [
                        "_matchScore",
                        "odiNumber",
                        "dateComplaintFiled",
                        "components",
                        "crash",
                        "fire",
                        "numberOfInjuries",
                        "numberOfDeaths",
                        "consumerLocation",
                        text_col,
                    ]
                    if c in out.columns
                ]
                st.dataframe(out[keep], use_container_width=True, hide_index=True)

    # --- Map ---
    with tabs[2]:
        if "stateAbbreviation" not in complaints_df.columns or complaints_df["stateAbbreviation"].dropna().empty:
            st.info("No complaint location data available from NHTSA.")
        else:
            geo = complaints_df.dropna(subset=["stateAbbreviation"]).copy()
            counts = geo["stateAbbreviation"].value_counts().rename_axis("state").reset_index(name="count")

            fig = px.choropleth(
                counts,
                locations="state",
                locationmode="USA-states",
                color="count",
                scope="usa",
                title="Complaints by state (from NHTSA consumer location)",
                color_continuous_scale="Blues",
            )

            fig.update_coloraxes(cmin=0, cmax=counts["count"].max())
            fig.update_geos(projection_type="albers usa", fitbounds=False)
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10), dragmode=False)

            config = {"scrollZoom": False, "displayModeBar": False, "doubleClick": False}
            st.plotly_chart(fig, use_container_width=True, config=config)

            st.dataframe(
                counts.sort_values("count", ascending=False).head(25),
                use_container_width=True,
                hide_index=True,
            )

    # --- Trends ---
    with tabs[3]:
        st.write("Complaint volume over time (by complaint filed date).")

        comp_df = component_frequency(complaints_df)
        components = ["All components"]
        if not comp_df.empty:
            components += comp_df["component"].tolist()

        selected = st.selectbox("Component", components, index=0)
        df_for_trend = complaints_df
        if selected != "All components" and "components" in complaints_df.columns:
            needle = f"|{selected}|"
            s = "|" + complaints_df["components"].fillna("").astype(str) + "|"
            df_for_trend = complaints_df[s.str.contains(needle, case=False, regex=False)]

        ts = complaints_time_series(df_for_trend, date_col="dateComplaintFiled")
        if ts.empty:
            st.info("No complaint dates available for this selection.")
        else:
            title = "Complaints per month" if selected == "All components" else f"Complaints per month — {selected}"
            fig = px.line(ts, x="month", y="count", markers=True, title=title)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "Lookup by VIN or Make/Model/Year in the sidebar and click **Analyze vehicle**. "
        "Double click cells to enlarge them."
    )
