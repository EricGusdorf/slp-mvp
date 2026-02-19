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

st.title("SLP Vehicle Defect MVP")
st.caption("NHTSA recalls + consumer complaints for quick defect pattern review.")

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
        make = st.text_input("Make", value="", placeholder="Honda")
        model = st.text_input("Model", value="", placeholder="Accord")
        year = st.number_input("Model year", min_value=1950, max_value=datetime.now().year + 1, value=2021, step=1)

    st.divider()
    enrich = st.checkbox(
        "Include complaint details (location + full text)",
        value=True,
        help="Needed for the map; improves symptom search. Uses a capped, cached NHTSA lookup per complaint.",
    )

    st.divider()
    analyze_clicked = st.button("Analyze vehicle", type="primary")


# Keep enrichment controls out of the UI (minimal MVP).
ENRICH_LIMIT = int(os.environ.get("SLP_ENRICH_LIMIT", "120"))
ENRICH_WORKERS = int(os.environ.get("SLP_ENRICH_WORKERS", "6"))


def _vehicle_from_vin(vin: str):
    decoded = decode_vin(vin, cache=cache)
    make = (decoded.get("Make") or "").strip()
    model = (decoded.get("Model") or "").strip()
    year = (decoded.get("ModelYear") or "").strip()
    warn = decoded.get("_vin_decode_warning")
    return make, model, int(year) if str(year).isdigit() else None, decoded, warn


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

            make, model, year, decoded, warn = _vehicle_from_vin(vin)
            if warn:
                st.warning(f"VIN decode warning: {warn}")
            if not make or not model or not year:
                st.error("Could not decode make/model/year from VIN. Try Make/Model/Year input.")
                st.stop()

            st.session_state["vehicle"] = {"make": make, "model": model, "year": year, "vin": vin, "decoded": decoded}

        else:
            if not make or not model or not year:
                st.error("Enter make, model, and year.")
                st.stop()
            st.session_state["vehicle"] = {"make": make, "model": model, "year": int(year), "vin": None, "decoded": None}

        v = st.session_state["vehicle"]
        with st.spinner("Fetching NHTSA recalls + complaints..."):
            recalls = fetch_recalls_by_vehicle(v["make"], v["model"], v["year"], cache=cache)
            complaints = fetch_complaints_by_vehicle(v["make"], v["model"], v["year"], cache=cache)

        st.session_state["raw_recalls"] = recalls
        st.session_state["raw_complaints"] = complaints

        recalls_df = recalls_to_df(recalls)
        complaints_df = complaints_to_df(complaints)

        enrich_stats = {"requested": 0, "enriched": 0, "failed": 0}
        if enrich and not complaints_df.empty:
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

    st.subheader(f"{v['year']} {v['make']} {v['model']}")
    if v.get("vin"):
        st.caption(f"VIN: `{v['vin']}`")
    # KPIs row
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
                fig = px.bar(top_n, x="count", y="component", orientation="h", title="Top complaint components")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_n[["component", "count", "share"]], use_container_width=True, hide_index=True)

        with right:
            st.subheader("Recalls")
            if recalls_df is None or recalls_df.empty:
                st.info("No recalls returned by NHTSA recallsByVehicle.")
            else:
                cols = [c for c in ["NHTSACampaignNumber", "Component", "Summary", "ReportReceivedDate"] if c in recalls_df.columns]
                st.dataframe(recalls_df[cols].head(50), use_container_width=True, hide_index=True)

        with st.expander("View complaints (first 100)"):
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
                st.dataframe(complaints_df[cols].head(100), use_container_width=True, hide_index=True)

        if enrich:
            st.caption(
                f"Enrichment (capped): requested {enrich_stats.get('requested',0)}, "
                f"enriched {enrich_stats.get('enriched',0)}, failed {enrich_stats.get('failed',0)}."
            )

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
            st.warning(
                "No complaint location data available. Enable 'Include complaint details (location + full text)' and re-run."
            )
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
            )
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(counts.sort_values("count", ascending=False).head(25), use_container_width=True, hide_index=True)

    # --- Trends ---
    with tabs[3]:
        st.write("Complaint volume over time (by complaint filed date).")

        # Filter by component to support make/model/component trend investigation.
        comp_df = component_frequency(complaints_df)
        components = ["All components"]
        if not comp_df.empty:
            components += comp_df["component"].tolist()

        selected = st.selectbox("Component", components, index=0)
        df_for_trend = complaints_df
        if selected != "All components" and "components" in complaints_df.columns:
            # Match whole component tokens in the pipe-delimited components field
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
