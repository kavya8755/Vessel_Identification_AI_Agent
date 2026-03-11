# Vessel Identification AI POC

This repository contains a **design-oriented proof of concept** for a vessel identification and search system, built around the provided `docs/case_study.pdf` and `docs/dataset.csv`.

The goal is to:
- Clean and interpret noisy vessel data.
- Demonstrate **identity resolution** across IMO / MMSI / name.
- Provide simple **search & retrieval** over structured data.
- Sketch how a **conversational AI layer** and **caching / sessions** could sit on top.

The POC is intentionally small and focused, using a single Python script plus a short design document.

---

## Quickstart

### 1. Create and activate a virtual environment (optional but recommended)

```bash
cd Vessel_Identification_AI_Agent
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the POC script

The main entrypoint is `poc_vessel_identity.py`. It expects `docs/dataset.csv` to be present.

#### a) Show basic dataset summary

```bash
python poc_vessel_identity.py summary
```

#### b) Run simple search with filters

Examples:

```bash
# Search by IMO
python poc_vessel_identity.py search --imo 9528574

# Search for chemical tankers by flag and minimum deadweight
python poc_vessel_identity.py search --vessel-type "Chemical Tanker" --flag SG --min-deadweight 30000

# Search by partial vessel name (case-insensitive)
python poc_vessel_identity.py search --name-contains "HOEGH"
```

#### c) Run identity resolution demo

This groups records into identity clusters using simple heuristics and shows conflicts.

```bash
python poc_vessel_identity.py resolve
```

#### d) Try the simple conversational shell

This is a lightweight REPL that simulates how a conversational AI agent might translate natural-ish language into structured filters and apply caching / session context.

```bash
python poc_vessel_identity.py chat
```

Example prompts inside the shell:

- `show tankers flag=SG deadweight>30000`
- `find vessels name~HOEGH`
- `repeat last but flag=LR`
- `clear cache`
- `exit`

---

## Files

- `docs/dataset.csv` – sample vessel dataset used for exploration, cleaning, and search.
- `poc_vessel_identity.py` – main Python POC script:
  - Loads and cleans the dataset.
  - Performs basic identity resolution (IMO/MMSI/name-based clustering).
  - Provides CLI commands for summary, search, resolve, and chat.
- `docs/design.md` – short design document capturing:
  - Data understanding and quality issues.
  - Identity resolution strategy.
  - High-level system design (ingestion, storage, processing, search).
  - How a conversational AI + caching/session layer could work.

---

