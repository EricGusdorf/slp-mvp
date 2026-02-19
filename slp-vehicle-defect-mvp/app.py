import os
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
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

# --- Sidebar ---
with st.sidebar:
    st.header("Vehicle input")
    input_mode = st.radio("Lookup by", ["VIN", "Make / Model / Year"])

    vin = ""
    make = ""
    model = ""
    year: Optional[int] = None

    if input_mode == "VIN":
        vin = st.text_input("VIN (17 chars)", placeholder="1HGCV1F56MA123456")
    else:
        make = st.text_input("Make", placeholder="Honda")
        model = st.text_input("Model", placeholder="Accord")
        year = st.number_input("Model year", min_value=1950, max_value=datetime.now().year + 1, value=2021)

    st.divider()
    enrich = st.checkbox("Include complaint details (location + full text)", value=True)
    st.divider()
    analyze_clicked = st.button("Analyze vehicle", type="primary")


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


# --- Analysis ---
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
                st.error("Could not decode make/model/year from VIN.")
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

        if recalls_err and complaints_err:
            st.error(
                "NHTSA services are currently unavailable. "
                "Please try again and confirm make, model, and year are correct."
            )
            st.stop()

        if (recalls_err is None) and (complaints_err is None) and (not recalls) and (not complaints):
            st.error(
                f"No NHTSA data found for {v['year']} {v['make']} {v['model']}. "
                "Verify spelling."
            )
            st.stop()

        recalls_df = recalls_to_df(recalls)
        complaints_df = complaints_to_df(complaints)

        if enrich and not complaints_df.empty:
            complaints_df, _ = enrich_complaints_df(
                complaints_df,
                cache=cache,
                max_records=int(ENRICH_LIMIT),
                max_workers=int(ENRICH_WORKERS),
            )

        st.session_state["recalls_df"] = recalls_df
        st.session_state["complaints_df"] = complaints_df

    except Exception as e:
        st.error(str(e))
        st.stop()


# --- Display ---
if "vehicle" in st.session_state:
    v = st.session_state["vehicle"]
    recalls_df = st.session_state.get("recalls_df", pd.DataFrame())
    complaints_df = st.session_state.get("complaints_df", pd.DataFrame())

    st.subheader(f"{v['year']} {v['make']} {v['model']}")

    sev = severity_summary(complaints_df)
    cols = st.columns(6)
    cols[0].metric("Complaints", len(complaints_df))
    cols[1].metric("Recalls", len(recalls_df))
    cols[2].metric("Crashes", sev["crash"])
    cols[3].metric("Fires", sev["fire"])
    cols[4].metric("Injuries", sev["injuries"])
    cols[5].metric("Deaths", sev["deaths"])

    tabs = st.tabs(["Summary", "Search", "Map", "Trends"])

    with tabs[0]:
        comp_df = component_frequency(complaints_df)
        if not comp_df.empty:
            fig = px.bar(comp_df.head(10), x="count", y="component", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        text_col = _best_text_column(complaints_df)
        query = st.text_input("Symptom query")
        if query and not complaints_df.empty:
            texts = complaints_df[text_col].fillna("").astype(str).tolist()
            idx = build_index(texts)
            matches = search_index(query, idx, top_k=10)
            rows = []
            for i, score_ in matches:
                row = complaints_df.iloc[i].to_dict()
                row["_score"] = round(score_, 3)
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tabs[2]:
        if "stateAbbreviation" in complaints_df.columns:
            geo = complaints_df.dropna(subset=["stateAbbreviation"])
            counts = geo["stateAbbreviation"].value_counts().rename_axis("state").reset_index(name="count")

            fig = px.choropleth(
                counts,
                locations="state",
                locationmode="USA-states",
                color="count",
                scope="usa",
                color_continuous_scale="Blues",  # Blue gradient
                title="Complaints by State",
            )

            fig.update_coloraxes(cmin=0, cmax=counts["count"].max())
            fig.update_geos(projection_type="albers usa")
            fig.update_layout(height=500, dragmode=False)

            config = {
                "scrollZoom": False,
                "displayModeBar": False,
            }

            st.plotly_chart(fig, use_container_width=True, config=config)

    with tabs[3]:
        ts = complaints_time_series(complaints_df, date_col="dateComplaintFiled")
        if not ts.empty:
            fig = px.line(ts, x="month", y="count", markers=True)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a VIN or make/model/year and click Analyze vehicle.")
