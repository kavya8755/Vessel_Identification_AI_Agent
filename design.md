## Vessel Identification & AI Search – Design POC

This document follows the structure and goals outlined in `case_study.pdf` and describes a **small, runnable POC** implemented in `poc_vessel_identity.py`.

---

## 1. Data understanding & quality issues

The dataset (`docs/dataset.csv`) contains both:
- **Static vessel attributes** – e.g. `imo`, `mmsi`, `name`, `vessel_type`, `flag`, `length`, `width`, tonnage fields, `builtYear`, `shipBuilder`, `propulsionType`, etc.
- **Dynamic AIS-like attributes** – e.g. `last_position_latitude`, `last_position_longitude`, `last_position_speed`, `destination`, `eta`, `matchedPort_*`, `staticData_updateTimestamp`, `last_position_updateTimestamp`.

Key identifiers:
- **IMO** – expected to be a stable global identifier, but we see issues:
  - Some rows with `imo = 0` or `imo = 1000000` (likely placeholders / invalid).
- **MMSI** – used by radios/transponders, but may:
  - Change over time for the same vessel.
  - Be recycled or misused, leading to conflicts.
- **Name / callsign / flag** – descriptive but mutable attributes; may be missing or noisy.

Observed quality issues from the sample:
- Placeholder or clearly invalid values:
  - `imo` equal to `0`, `1000000`, or other unlikely values.
  - Names like `"00000000000000000000"`.
  - Empty names or missing `vessel_type` / `flag`.
- Multiple rows where:
  - **Same IMO appears with different MMSIs** (possible MMSI changes or data errors).
  - **Same MMSI appears with different IMOs** (re-use, misconfiguration, or merge errors).
- Free-text inconsistencies:
  - Destinations like `"FOR ORDER"`, `"FISHINGGROUNDS"`, `"/???0MVAAT"`, etc.
  - Mixed casing and punctuation in names and destinations.

---

## 2. Identity resolution approach (POC level)

The POC focuses on simple, explainable heuristics rather than ML:

### 2.1 Canonical vessel key (cluster ID)

We treat each input row as a node in an **identity graph**:

- Two records are **linked** if any of the following hold:
  - Same non-placeholder `imo`.
  - Same `mmsi` and similar `name` (case-insensitive exact match or close match).
  - Same `name` and `flag` and similar dimensions (`length`, `width`) within a small tolerance.

We then compute **connected components** of this graph. Each component becomes a **vessel identity cluster** with a generated `vessel_id`:

\[
\text{vessel\_id} = \text{"V"} + \text{min(imo in cluster or synthetic index)}
\]

For the POC, this clustering is implemented using:
- A simple union-find (disjoint-set) or
- Pandas group-by passes over `imo`, `mmsi`, and (`name`, `flag`) pairs with heuristics.

### 2.2 Placeholder / invalid IMO detection

We flag records as **invalid IMO** when:
- `imo` is `0` or negative.
- `imo` is in a small, curated list of known placeholders (e.g. `1000000`, `8000000` from the sample).

These invalid IMO values are:
- Not used as canonical identifiers.
- Still used as secondary signals (combined with name/flag) if necessary, but down-weighted.

### 2.3 Conflict detection

Within each identity cluster we compute:
- Distinct IMOs and MMSIs.
- Distinct names and flags.

We then flag clusters as:
- **Clean** – one IMO, possibly multiple MMSIs over time.
- **MMSI conflict** – one IMO, many MMSIs (beyond a small threshold).
- **IMO conflict** – one MMSI, many IMOs.
- **Attribute drift** – large variation in name or flag over time.

This is implemented in the POC as simple aggregations over clusters.

---

## 3. High-level system design

### 3.1 Architecture overview

At a high level, a production system could look like:

```text
                 +--------------------------+
                 | External Data Sources    |
                 | - AIS feeds             |
                 | - Registry/port data    |
                 +------------+-------------+
                              |
                              v
                  +-----------+-----------+
                  | Ingestion & Staging   |
                  | - Batch / streaming   |
                  | - Schema validation   |
                  | - Basic cleaning      |
                  +-----------+-----------+
                              |
                              v
                  +-----------+-----------+
                  | Identity Resolution   |
                  | - Rules + ML models  |
                  | - Graph / clustering |
                  +-----------+-----------+
                              |
                              v
         +--------------------+--------------------+
         |                                         |
         v                                         v
+--------+---------+                    +----------+---------+
| Ground Truth DB  |                    | Search Index (OLAP)|
| (OLTP, normalized)|                   | (e.g. Elastic/OLAP)|
+--------+---------+                    +----------+---------+
         |                                         |
         +--------------------+--------------------+
                              v
                    +---------+---------+
                    | API & Query Layer |
                    | - REST/GraphQL   |
                    | - gRPC           |
                    +---------+---------+
                              |
                              v
                 +------------+------------+
                 | Conversational AI Layer |
                 | - LLM + tools          |
                 | - Session & caching    |
                 +------------------------+
```

For the POC we reduce this to:
- A **single Python process** with:
  - Pandas DataFrame as in-memory table.
  - Heuristic identity resolution.
  - Simple search functions.
  - A CLI / REPL representing the API and conversational layer.

---

## 4. Tooling choices

### 4.1 POC choices

- **Language**: Python (per instructions).
- **Data processing**: `pandas` for tabular manipulation.
- **CLI / demonstration**: standard library `argparse` plus simple REPL for the "chat" mode.
- **Tabular display**: `tabulate` for readable console tables.

### 4.2 Reasonable production options (not fully implemented)

- **Databases**:
  - **OLTP**: PostgreSQL or Aurora for normalized vessel / identity data.
  - **Graph store**: Neo4j, JanusGraph, or Postgres + graph extension for identity relationships.
  - **Search index**: OpenSearch / Elasticsearch for text + geo + filter queries.
- **AI / LLM**:
  - Hosted LLM (OpenAI, Anthropic, or similar) or self-hosted model.
  - Embedding model for semantic search over text fields (e.g. vessel descriptions, documents).
- **Streaming / ingestion**:
  - Kafka / Kinesis / PubSub for AIS streams.
  - Scheduled batch ingestion for registry updates.

---

## 5. Conversational AI & tool-calling

In production, the conversational AI layer would:

1. **Parse user intent** (e.g. "Show all LNG carriers headed to Singapore with ETA before tomorrow").
2. **Ground the query** in the schema:
   - Map "LNG carriers" → `vessel_type == "LNG Carrier"`.
   - Map "Singapore" → port UNLOCODE(s) or geo box around Singapore.
   - Map "before tomorrow" → time range on `eta`.
3. **Call tools / APIs**:
   - Structured search endpoint on the vessel DB / search index.
   - Possibly an identity-explainer endpoint ("Why do you think these records are the same vessel?").
4. **Post-process** results:
   - Summarize.
   - Highlight conflicts or uncertainty.
   - Present links to drill down.

In the POC:
- We implement a **deterministic parser** for very simple patterns:
  - `key=value`, `key>number`, `key<number`, `name~SUBSTRING`.
  - Supported keys: `imo`, `mmsi`, `name`, `flag`, `vessel_type`, `deadweight`, `grossTonnage`.
- These are mapped to pandas filters and executed.
- The REPL simulates "chat" by:
  - Keeping track of **last filters** (session context).
  - Allowing modifications like "repeat last but flag=SG".

---

## 6. Caching & session management

### 6.1 Goals

- Avoid re-running expensive queries repeatedly within a session.
- Maintain conversational context so the user can refine queries.

### 6.2 POC implementation

In `poc_vessel_identity.py`:
- We maintain an **in-memory cache**:
  - Key: normalized representation of the structured filters (e.g. a sorted tuple).
  - Value: DataFrame slice (or indexes) for that query.
- The chat REPL:
  - Stores the **last successful filter set** as `session_state.last_filters`.
  - The user can type natural-ish shortcuts:
    - `repeat last` → reuse last filters.
    - `repeat last but flag=LR` → start from last, override `flag`.
  - When the same filters are seen again, results are served from the cache.

In production, this could be backed by:
- Redis / Memcached for shared caching across API servers.
- Session IDs passed via headers / tokens, with:
  - Short TTL caches for query results.
  - Lightweight storage of "last queries" for each session.

---

## 7. Evaluation strategies

Even at POC level, we can outline evaluation ideas:

- **Identity resolution**:
  - Build a small labeled test set of known vessel identities and conflicts.
  - Compute precision/recall for:
    - "Same vessel" predictions.
    - "Conflict" flags.
- **Search quality**:
  - Define sample queries with expected result sets.
  - Measure:
    - Whether expected vessels are present in top-N.
    - Ordering heuristics (e.g. by recency or relevance).
- **Conversational AI**:
  - Offline evaluation:
    - Prompt → structured query translation correctness.
  - Online / UX evaluation:
    - Time-to-answer for analysts.
    - Number of follow-up clarifications needed.
- **System performance**:
  - Latency and throughput of search endpoints.
  - Cache hit ratio and impact on latency.

---

## 8. How this POC maps to the case study questions

- **How can you determine when two records refer to the same vessel?**
  - Use a combination of stable identifiers (IMO), semi-stable identifiers (MMSI), and fuzzy attributes (name, flag, dimensions), aggregated via clustering.

- **How would you detect and flag invalid or conflicting records?**
  - Rule-based detection of invalid IMOs and placeholder values.
  - Cluster-level analysis of multiple IMOs/MMSIs/flags/names.

- **How could you track a vessel’s changes over time?**
  - Store time-stamped events per identity cluster; in the dataset we can use update timestamps and sort per vessel.

- **Is it realistic to create a “ground truth” vessel database? How?**
  - Yes, but only probabilistically:
    - Combine multiple sources, apply identity resolution, maintain explicit confidence scores and audit trails.

- **What system design would you propose for a search & retrieval solution?**
  - See the architecture diagram: ingestion → identity resolution → ground truth DB + search index → API → conversational AI.

- **How could a conversational AI interface support vessel search?**
  - Translate natural language into structured filters and tool calls, surface explanations, and enable iterative refinement.

- **How would you prevent LLM hallucinations when answering vessel-related queries?**
  - Force the LLM to:
    - Only answer from **retrieved, structured data**.
    - Explicitly say "I don’t know" when the DB has no matching records.
    - Log and audit all underlying queries and results.

- **What evaluation methods would you use?**
  - See Section 7: labeled test sets, search quality metrics, UX studies, and system performance metrics.

