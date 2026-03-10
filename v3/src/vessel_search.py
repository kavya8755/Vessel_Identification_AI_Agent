"""
Vessel Search & Retrieval Engine
==================================
Provides structured search, fuzzy name matching, and filter-based querying
over the cleaned vessel dataset — ready to be called by an AI agent.
"""

import pandas as pd
import numpy as np
from typing import Optional, Any
import re
import json
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: simple TF-style fuzzy scorer
# ---------------------------------------------------------------------------
def fuzzy_score(query: str, target: str) -> float:
    """Returns a 0-1 similarity between two strings (case-insensitive)."""
    q, t = query.lower().strip(), target.lower().strip()
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t:
        return 0.9
    return SequenceMatcher(None, q, t).ratio()


# ---------------------------------------------------------------------------
# VesselSearchEngine
# ---------------------------------------------------------------------------
class VesselSearchEngine:
    """
    In-memory search engine on top of the resolved vessel DataFrame.

    Supports:
    - Exact IMO / MMSI lookup
    - Fuzzy name search
    - Multi-attribute filter queries (vessel_type, flag, built_year range, etc.)
    - Combined semantic + filter search (used by the AI agent)
    """

    SEARCHABLE_FILTERS = {
        "vessel_type": str,
        "flag": str,
        "aisClass": str,
        "propulsionType": str,
        "shipBuilder": str,
    }
    RANGE_FILTERS = {
        "builtYear": float,
        "deadweight": float,
        "grossTonnage": float,
        "length": float,
        "speed": float,      # maps to last_position_speed
    }

    def __init__(self, resolved_df: pd.DataFrame):
        self.df = resolved_df.copy()
        self._preprocess()

    def _preprocess(self):
        """Normalise key columns for search."""
        self.df["_name_search"] = self.df["name"].fillna("").str.upper().str.strip()
        self.df["_flag_upper"] = self.df["flag"].fillna("").str.upper().str.strip()
        self.df["_type_lower"] = self.df["vessel_type"].fillna("").str.lower().str.strip()
        # Alias speed column
        if "last_position_speed" in self.df.columns:
            self.df["speed"] = self.df["last_position_speed"]

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------
    def lookup_by_imo(self, imo: int) -> pd.DataFrame:
        return self.df[self.df["imo"] == imo]

    def lookup_by_mmsi(self, mmsi: int) -> pd.DataFrame:
        return self.df[self.df["mmsi"] == mmsi]

    def search_by_name(self, name: str, top_k: int = 10) -> pd.DataFrame:
        """Fuzzy name search. Returns top_k best matches."""
        q = name.upper().strip()
        scores = self.df["_name_search"].apply(lambda t: fuzzy_score(q, t))
        idx = scores.nlargest(top_k).index
        result = self.df.loc[idx].copy()
        result["_score"] = scores[idx]
        return result.sort_values("_score", ascending=False)

    def filter_query(self, filters: dict[str, Any], top_k: int = 50) -> pd.DataFrame:
        """
        Apply structured filters. `filters` dict supports:
          - vessel_type, flag, aisClass, propulsionType, shipBuilder  (exact/substring match)
          - builtYear_min, builtYear_max
          - deadweight_min, deadweight_max
          - grossTonnage_min, grossTonnage_max
          - length_min, length_max
          - name  (fuzzy)
          - imo, mmsi (exact)
        """
        result = self.df.copy()

        # Exact lookups
        if "imo" in filters:
            result = result[result["imo"] == int(filters["imo"])]
        if "mmsi" in filters:
            result = result[result["mmsi"] == int(filters["mmsi"])]

        # Text / categorical filters (case-insensitive substring)
        for col in self.SEARCHABLE_FILTERS:
            if col in filters:
                val = str(filters[col]).lower().strip()
                result = result[result[col].fillna("").str.lower().str.contains(val, na=False)]

        # Range filters
        for base_col in self.RANGE_FILTERS:
            col = base_col if base_col in result.columns else None
            if col is None:
                continue
            if f"{base_col}_min" in filters:
                result = result[result[col].fillna(-np.inf) >= float(filters[f"{base_col}_min"])]
            if f"{base_col}_max" in filters:
                result = result[result[col].fillna(np.inf) <= float(filters[f"{base_col}_max"])]

        # Fuzzy name on top of other filters
        if "name" in filters:
            q = str(filters["name"]).upper().strip()
            scores = result["_name_search"].apply(lambda t: fuzzy_score(q, t))
            result = result[scores > 0.4].copy()
            result["_score"] = scores[scores > 0.4]
            result = result.sort_values("_score", ascending=False)

        return result.head(top_k)

    def combined_search(self, query: str, filters: Optional[dict] = None, top_k: int = 10) -> pd.DataFrame:
        """
        Semantic-style search: parse free-text query for field hints,
        combine with explicit filters.
        """
        parsed = self._parse_free_text(query)
        merged = {**parsed, **(filters or {})}
        if merged:
            return self.filter_query(merged, top_k=top_k)
        # Fall back to name search
        return self.search_by_name(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal: parse free-text into filter dict
    # ------------------------------------------------------------------
    def _parse_free_text(self, text: str) -> dict:
        parsed = {}
        # IMO / MMSI patterns
        imo_m = re.search(r"\bIMO[:\s#]*(\d{7})\b", text, re.I)
        mmsi_m = re.search(r"\bMMSI[:\s#]*(\d{9})\b", text, re.I)
        if imo_m:
            parsed["imo"] = int(imo_m.group(1))
        if mmsi_m:
            parsed["mmsi"] = int(mmsi_m.group(1))

        # Year patterns
        year_m = re.search(r"\bbuilt\s+in\s+(\d{4})\b", text, re.I)
        if year_m:
            parsed["builtYear_min"] = int(year_m.group(1))
            parsed["builtYear_max"] = int(year_m.group(1))

        # Vessel type keywords mapping
        type_map = {
            "tanker": "tanker", "bulk": "bulk", "container": "container",
            "lng": "lng", "tug": "tug", "fishing": "fishing", "passenger": "passenger",
            "cargo": "cargo", "ro-ro": "ro-ro", "reefer": "reefer",
        }
        for kw, vtype in type_map.items():
            if re.search(rf"\b{kw}\b", text, re.I):
                parsed["vessel_type"] = vtype
                break

        # Flag / country patterns (2-letter ISO codes)
        flag_m = re.search(r"\bflag[:\s]+([A-Z]{2})\b", text, re.I)
        if flag_m:
            parsed["flag"] = flag_m.group(1).upper()

        return parsed

    # ------------------------------------------------------------------
    # Format results for LLM consumption
    # ------------------------------------------------------------------
    def format_results(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Return a concise JSON-serialisable string for the LLM."""
        cols = ["vessel_id", "imo", "mmsi", "name", "vessel_type", "flag",
                "builtYear", "deadweight", "grossTonnage",
                "last_position_latitude", "last_position_longitude",
                "last_position_speed", "destination", "matchedPort_name"]
        existing_cols = [c for c in cols if c in df.columns]
        subset = df[existing_cols].head(max_rows).fillna("N/A")
        return subset.to_json(orient="records", indent=2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from vessel_identity import VesselIdentityResolver

    raw = pd.read_csv("../data/dataset.csv", low_memory=False)
    resolver = VesselIdentityResolver(raw)
    resolver.parse_records()
    resolver.validate_identifiers()
    resolver.detect_conflicts()
    resolved = resolver.resolve_identities()

    engine = VesselSearchEngine(resolved)

    print("=== Fuzzy name search: 'MARCO' ===")
    r = engine.search_by_name("MARCO", top_k=3)
    print(r[["name", "imo", "mmsi", "vessel_type", "flag", "_score"]].to_string(index=False))

    print("\n=== Filter: Dry Bulk vessels, flag=LR ===")
    r2 = engine.filter_query({"vessel_type": "Dry Bulk", "flag": "LR"}, top_k=5)
    print(r2[["name", "imo", "mmsi", "builtYear", "deadweight"]].to_string(index=False))

    print("\n=== Combined search: 'container ship built in 2023' ===")
    r3 = engine.combined_search("container ship built in 2023", top_k=5)
    print(r3[["name", "imo", "builtYear", "vessel_type"]].to_string(index=False))
