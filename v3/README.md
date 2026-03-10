# Vessel Identification & AI Agent

A complete solution for vessel identity resolution, data quality analysis,
and AI-powered conversational search over maritime AIS data.

## Project Structure

```
vessel_project/
├── src/
│   ├── vessel_identity.py   # IMO/MMSI validation, deduplication, change tracking
│   ├── vessel_search.py     # Filter-based + fuzzy name search engine
│   └── vessel_agent.py      # Conversational AI agent with caching & session mgmt
├── notebooks/
│   └── eda.py               # Dataset EDA and statistics script
├── data/
│   └── dataset.csv          # Source vessel data (1,734 records)
└── docs/
    └── design_document.docx # Full design document with diagrams & tables
```

## Quick Start

```bash
# Run identity resolution
cd src && python vessel_identity.py

# Run search engine demo
cd src && python vessel_search.py

# Run conversational agent (set GROQ_API_KEY for live LLM responses)
export GROQ_API_KEY=your_key_here
cd src && python vessel_agent.py

# Run EDA
cd notebooks && python eda.py
```

## Key Findings from the Dataset

- **1,734** raw records → **1,338** unique vessels after resolution
- **697** records (40.2%) have identifier issues (fake/invalid IMO)
- **196** conflicts detected (all one-IMO → many-MMSI type)
- Most common fake IMO: `1000000` (21 occurrences)
- Worst MMSI churn: IMO `9710749` mapped to 28 different MMSIs

## Architecture

```
User Query
    │
    ▼
IntentParser  ──► structured filters
    │
    ▼
VesselSearchEngine  ──► retrieved records (ground truth)
    │
    ▼
QueryCache (LRU, 5 min TTL)
    │
    ▼
LLM Prompt (retrieved records injected as context)
    │
    ▼
Claude API  ──► grounded answer (no hallucinations)
    │
    ▼
SessionManager (appends to history)
```

## Anti-Hallucination Design

The agent uses a strict RAG (Retrieval-Augmented Generation) pattern:
- Retrieved vessel records are injected verbatim into every LLM prompt
- System prompt prohibits the LLM from inventing vessel details
- If data is missing, the agent explicitly says so
- Position data always includes the `last_position_updateTimestamp`

## Requirements

```
pandas
numpy
# Optional (for live LLM):
anthropic
```
