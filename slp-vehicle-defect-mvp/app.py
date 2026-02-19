import os
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from slp_mvp.cache import DiskCache
from slp_mvp.nhtsa import decode_vin, fetch_complaints_by_vehicle, fetch_recalls_by_vehicle, NHTSAError
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

        # Common makes (prioritized A–Z)
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

        # Match case-insensitively against vPIC results, but return the vPIC spelling
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


st.set_page_config(page_title="SLP Vehicle Defect MVP", layout="wide")

st.markdown(
    """
    <div style="
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    ">
        SLP Vehicle Defect MVP
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Cache ---
DEFAULT_CACHE_DIR = os.environ.get("SLP_CACHE_DIR", ".cache")
cache = DiskCache(DEFAULT_CACHE_DIR)

# --- Sidebar: vehicle selection ---
with st.sidebar:
    st.header("Vehicle input")
    input_mode = st.radio("Lookup by", ["VIN", "Make / Model / Year"], horizontal=False)

    vin = ""
    make = ""
    model = ""
    year: Optional[int] = None

    if input_mode == "VIN":
        vin = st.text_input("VIN (17 chars)", value="", placeholder="e.g., 1HGCV1F56MA123456")
    else:
        year = st.number_input(
            "Model year",
            min_value=1950,
            max_value=datetime.now().year + 1,
            value=2021,
            step=1,
        )

        makes = vp_get_all_makes()

        if "make_index" not in st.session_state:
            st.session_state.make_index = 0

        make = st.selectbox(
            "Make",
            options=[""] + makes,
            index=st.session_state.make_index,
            key="make_selectbox",
        )

        # Persist selected index
        if make != "":
            st.session_state.make_index = ([""] + makes).index(make)

        models: list[str] = []
        if make:
            models = vp_get_models_for_make_year(make, int(year))

        model = st.selectbox("Model", options=[""] + models, index=0, disabled=(not make))

    st.divider()
    analyze_clicked = st.button("Analyze vehicle", type="primary")

# Enrichment always on (no UI toggle)
enrich = True

# Keep enrichment controls out of the UI (minimal MVP).
ENRICH_LIMIT = int(os.environ.get("SLP_ENRICH_LIMIT", "120"))
ENRICH_WORKERS = int(os.environ.get("SLP_ENRICH_WORKERS", "6"))


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
        if input_mode == "VIN":
            if not vin:
                st.error("Enter a VIN.")
                st.stop()

            make_, model_, year_, decoded, warn = _vehicle_from_vin(vin)
            if warn:
                st.warning(f"VIN decode warning: {warn}")
            if not make_ or not model_ or not year_:
                st.error("Could not decode make/model/year from VIN. Try Make/Model/Year input.")
                st.stop()

            st.session_state["vehicle"] = {
                "make": make_,
                "model": model_,
                "year": year_,
                "vin": vin,
                "decoded": decoded,
            }

        else:
            if not make or not model or not year:
                st.error("Enter make, model, and year.")
                st.stop()

            st.session_state["vehicle"] = {
                "make": make,
                "model": model,
                "year": int(year),
                "vin": None,
                "decoded": None,
            }

        v = st.session_state["vehicle"]

        # Fetch: handle per-endpoint errors so we can distinguish
        # "invalid vehicle" (empty data) vs "NHTSA unavailable" (errors).
        with st.spinner("Fetching NHTSA recalls + complaints..."):
            recalls = []
            complaints = []
            recalls_err = None
            complaints_err = None

            try:
                recalls = fetch_recalls_by_vehicle(v["make"], v["model"], v["year"], cache=cache)
            except NHTSAError as e:
                recalls_err = str(e)

            try:
                complaints = fetch_complaints_by_vehicle(v["make"], v["model"], v["year"], cache=cache)
            except NHTSAError as e:
                complaints_err = str(e)

        # If BOTH endpoints failed, show a service error (not an invalid vehicle).
        if recalls_err and complaints_err:
            st.error(
                "NHTSA services are currently unavailable for this lookup. "
                "Please try again and confirm the vehicle make, model, and year are spelled correctly."
            )
            with st.expander("Details"):
                st.code(f"recalls error:\n{recalls_err}\n\ncomplaints error:\n{complaints_err}")
            st.stop()

        # If requests succeeded but returned no data, treat as invalid vehicle.
        if (recalls_err is None) and (complaints_err is None) and (not recalls) and (not complaints):
            st.error(
                f"No NHTSA data found for {v['year']} {v['make']} {v['model']}. "
                "Verify the make, model, and year."
            )
            st.stop()

        # Partial failure: continue with warning (minimal MVP).
        if recalls_err and not complaints_err:
            st.warning("Recalls lookup failed; showing complaints only. Vehicle may not exist")
        if complaints_err and not recalls_err:
            st.warning("Complaints lookup failed; showing recalls only.")

        st.session_state["raw_recalls"] = recalls
        st.session_state["raw_complaints"] = complaints

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

        st.session_state["recalls_df"] = recalls_df
        st.session_state["complaints_df"] = complaints_df
        st.session_state["enrich_stats"] = enrich_stats

    except NHTSAError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()


# --- Display results if available ---
if "vehicle" in st.session_state:
    v = st.session_state["vehicle"]
    recalls_df = st.session_state.get("recalls_df", pd.DataFrame())
    complaints_df = st.session_state.get("complaints_df", pd.DataFrame())
    enrich_stats = st.session_state.get("enrich_stats", {"requested": 0, "enriched": 0, "failed": 0})

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
            st.subheader("Defect patterns")
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
    
                # Rename and format columns for clarity
                display_df = top_n[["component", "count", "share"]].copy()
                display_df = display_df.rename(columns={
                    "component": "Component",
                    "count": "Complaint Count",
                    "share": "Share of Total Complaints (%)",
                })
    
                display_df["Share of Total Complaints (%)"] = (
                    display_df["Share of Total Complaints (%)"] * 100
                ).round(1)
    
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
        with right:
            st.subheader("Recalls")
            if recalls_df is None or recalls_df.empty:
                st.info("No recalls returned by NHTSA recallsByVehicle.")
            else:
                cols = [
                    c
                    for c in ["NHTSACampaignNumber", "Component", "Summary", "ReportReceivedDate"]
                    if c in recalls_df.columns
                ]
                st.dataframe(recalls_df[cols].head(50), use_container_width=True, hide_index=True)
    
        with st.expander("View complaints (all)"):
            if complaints_df is None or complaints_df.empty:
                st.info("No complaints returned by NHTSA complaintsByVehicle.")
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
            st.warning("No complaint location data available from NHTSA.")
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
    st.info("Enter a VIN or make/model/year in the sidebar and click **Analyze vehicle**.")
