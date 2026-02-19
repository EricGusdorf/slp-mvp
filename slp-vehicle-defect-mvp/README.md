# SLP Vehicle Defect Analyzer — MVP

A working prototype that helps attorneys quickly assess **vehicle defect patterns** and **case strength** using public NHTSA data (recalls + consumer complaints).

This repo is designed to satisfy the take‑home requirements:
- Surface relevant **recalls** + **consumer complaints**
- Identify defect **patterns** (components failing most often)
- Highlight **severity indicators** (crashes, fires, injuries, deaths)
- Search for **similar cases by symptom text**
- Provide **geographic context**
- Show **complaint volume trends over time**

---

## Quick start

### 1) Create a virtualenv
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the app
```bash
streamlit run app.py
```

The app will open in your browser (default `http://localhost:8501`).

---

## How to use

1. Choose **VIN** or **Make / Model / Year** in the left sidebar.
2. Click **Analyze vehicle**.
3. Explore the tabs:
   - **Summary**: defect patterns + recalls + a small complaints preview
   - **Search**: symptom text search within complaints
   - **Map**: complaints by state (requires enrichment)
   - **Trends**: complaints per month, optionally filtered by component

---

## Data sources (NHTSA)

This MVP uses three main public endpoints:

### Recalls (by vehicle)
`https://api.nhtsa.gov/recalls/recallsByVehicle?make={MAKE}&model={MODEL}&modelYear={YEAR}`

### Complaints (by vehicle)
`https://api.nhtsa.gov/complaints/complaintsByVehicle?make={MAKE}&model={MODEL}&modelYear={YEAR}`

### Complaint enrichment (location + full text)
To support geographic mapping and richer text search, the app optionally enriches complaints via:

`https://api.nhtsa.gov/safetyIssues/byNhtsaId?filter=issueType&filterValue=complaints&nhtsaId={ODI_NUMBER}`

This returns `consumerLocation` (e.g., `"LAS VEGAS, NV"`) and a longer complaint `description`, plus structured component labels.

### VIN decoding (vPIC)
VINs are decoded with the vPIC API:
`https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{VIN}?format=json`

---

## Architecture

**Single Streamlit app** (no backend server).

```
app.py (UI)
  |
  +-- slp_mvp/nhtsa.py       (NHTSA + vPIC API calls)
  +-- slp_mvp/cache.py       (disk JSON cache; URL-keyed)
  +-- slp_mvp/enrich.py      (parallel enrichment using safetyIssues/byNhtsaId)
  +-- slp_mvp/analytics.py   (dataframes, component frequency, trends)
  +-- slp_mvp/text_search.py (TF‑IDF search index)
```

### Caching
API responses are cached on disk in `.cache/` keyed by URL (SHA‑256 hash). This makes iterative investigation fast and reduces repeated API calls.

You can change the cache directory:
```bash
export SLP_CACHE_DIR=/tmp/slp_cache
```

---

## Case strength (MVP)

This MVP keeps case strength intentionally transparent by surfacing:
- Recall presence
- Complaint volume
- Severity indicators (crash/fire/injury/death counts)
- Component concentration (top complaint components)
- Volume trends over time (optionally by component)

---

## Symptom search (MVP)

This MVP implements lexical semantic-ish search using **TF‑IDF + cosine similarity** over complaint text.
- If enrichment is enabled, search uses the longer `description` text.
- Otherwise it searches the shorter complaint `summary`.

Extensions (see below) include true embedding search.

---

## Tradeoffs / assumptions

- **Geography requires enrichment**: the `complaintsByVehicle` endpoint does not reliably include structured `consumerLocation`, so the MVP enriches each complaint via the `safetyIssues/byNhtsaId` endpoint (cached, capped, parallelized).
- **Search is TF‑IDF**: good enough for MVP symptom matching, but not as strong as embeddings for paraphrases.
- **No persistence beyond cache**: the app doesn’t maintain a full database; it is optimized for “investigate a vehicle quickly.”

---

## Extensions (if given another week)

- Ingest NHTSA complaint flat files (5‑year or full history) into SQLite/DuckDB for:
  - instant geography without per‑complaint API calls
  - cross‑vehicle comparisons and “similar vehicle families” searches
- Add true **embedding search** (local model or API) + component-aware reranking.
- Add manufacturer communications (TSBs), investigations, and recall remedy effectiveness.
- Add an “intake memo” export (PDF/Word) summarizing findings for the file.
- Add multi-vehicle dashboards (e.g., “top emerging issues across Honda 2019–2022”).

---

## AI tools used

ChatGPT 5.2 PRO

---

## Disclaimer

This prototype is for internal triage and product exploration only and does not constitute legal advice.
