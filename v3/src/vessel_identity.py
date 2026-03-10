"""
Vessel Identity Resolution Module
===================================
Handles deduplication, conflict detection, and identity resolution
for noisy vessel data using IMO and MMSI identifiers.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import hashlib
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_IMO_MIN = 1_000_000
VALID_IMO_MAX = 9_999_999
VALID_MMSI_MIN = 100_000_000
VALID_MMSI_MAX = 999_999_999
FAKE_IMO_VALUES = {0, 1_000_000, 3_395_388, 2_097_152, 4_194_336, 8_000_000}
FAKE_MMSI_VALUES = {0}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class VesselRecord:
    imo: int
    mmsi: int
    name: str
    vessel_type: str
    flag: str
    callsign: str
    ais_class: str
    static_update_ts: str
    raw_index: int
    issues: list = field(default_factory=list)
    resolved_identity: Optional[str] = None


@dataclass
class IdentityConflict:
    conflict_type: str           # 'one_imo_many_mmsi' | 'one_mmsi_many_imo' | 'duplicate'
    imo: Optional[int]
    mmsi: Optional[int]
    record_indices: list
    description: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def luhn_check_imo(imo: int) -> bool:
    """
    IMO numbers use a weighted check digit (digits 1-6 multiplied by 7-2 respectively).
    Returns True if the IMO passes the checksum.
    """
    s = str(imo).zfill(7)
    if len(s) != 7:
        return False
    total = sum(int(s[i]) * (7 - i) for i in range(6))
    return total % 10 == int(s[6])


def is_valid_imo(imo: int) -> bool:
    """Validates IMO: range check + checksum."""
    if imo in FAKE_IMO_VALUES:
        return False
    if not (VALID_IMO_MIN <= imo <= VALID_IMO_MAX):
        return False
    return luhn_check_imo(imo)


def is_valid_mmsi(mmsi: int) -> bool:
    """
    Basic MMSI validation:
    - 9 digits
    - Not in known fake values
    - Starts with valid MID (Maritime Identification Digit 2-7)
    """
    if mmsi in FAKE_MMSI_VALUES:
        return False
    s = str(mmsi)
    if len(s) != 9:
        return False
    first_digit = int(s[0])
    return 2 <= first_digit <= 7


def classify_imo(imo: int) -> str:
    """Return a human-readable classification of an IMO value."""
    if imo == 0:
        return "ZERO_IMO"
    if imo in FAKE_IMO_VALUES:
        return "KNOWN_FAKE"
    if not (VALID_IMO_MIN <= imo <= VALID_IMO_MAX):
        return "OUT_OF_RANGE"
    if not luhn_check_imo(imo):
        return "INVALID_CHECKSUM"
    return "VALID"


# ---------------------------------------------------------------------------
# Main Processor
# ---------------------------------------------------------------------------
class VesselIdentityResolver:
    """
    Processes a vessel DataFrame to:
    1. Validate IMO and MMSI identifiers
    2. Flag conflicts (one IMO → many MMSIs and vice versa)
    3. Resolve duplicate records
    4. Produce a cleaned 'ground-truth' table with unique vessel IDs
    """

    def __init__(self, df: pd.DataFrame):
        self.raw = df.copy()
        self.records: list[VesselRecord] = []
        self.conflicts: list[IdentityConflict] = []
        self.resolved_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Step 1 – Parse raw rows into VesselRecord objects
    # ------------------------------------------------------------------
    def parse_records(self) -> None:
        logger.info("Parsing %d raw records ...", len(self.raw))
        for idx, row in self.raw.iterrows():
            rec = VesselRecord(
                imo=int(row["imo"]) if pd.notna(row["imo"]) else 0,
                mmsi=int(row["mmsi"]) if pd.notna(row["mmsi"]) else 0,
                name=str(row.get("name", "")) or "",
                vessel_type=str(row.get("vessel_type", "")) or "",
                flag=str(row.get("flag", "")) or "",
                callsign=str(row.get("callsign", "")) or "",
                ais_class=str(row.get("aisClass", "")) or "",
                static_update_ts=str(row.get("staticData_updateTimestamp", "")),
                raw_index=idx,
            )
            self.records.append(rec)

    # ------------------------------------------------------------------
    # Step 2 – Validate identifiers
    # ------------------------------------------------------------------
    def validate_identifiers(self) -> None:
        logger.info("Validating identifiers ...")
        for rec in self.records:
            imo_class = classify_imo(rec.imo)
            if imo_class != "VALID":
                rec.issues.append(f"IMO_{imo_class}")

            if not is_valid_mmsi(rec.mmsi):
                rec.issues.append("MMSI_INVALID")

            # Suspicious name patterns
            if re.fullmatch(r"0+", rec.name) or rec.name in ("", "nan"):
                rec.issues.append("NAME_MISSING_OR_FAKE")
            if re.search(r"[^\x20-\x7E]", rec.name):
                rec.issues.append("NAME_NON_ASCII")

    # ------------------------------------------------------------------
    # Step 3 – Detect conflicts
    # ------------------------------------------------------------------
    def detect_conflicts(self) -> None:
        logger.info("Detecting conflicts ...")

        valid_recs = [r for r in self.records if not any("IMO_" in i for i in r.issues)]

        # one IMO → many MMSIs
        imo_to_mmsis: dict[int, set] = {}
        imo_to_idxs: dict[int, list] = {}
        for r in valid_recs:
            imo_to_mmsis.setdefault(r.imo, set()).add(r.mmsi)
            imo_to_idxs.setdefault(r.imo, []).append(r.raw_index)

        for imo, mmsis in imo_to_mmsis.items():
            if len(mmsis) > 1:
                self.conflicts.append(IdentityConflict(
                    conflict_type="one_imo_many_mmsi",
                    imo=imo,
                    mmsi=None,
                    record_indices=imo_to_idxs[imo],
                    description=f"IMO {imo} maps to {len(mmsis)} MMSIs: {mmsis}",
                ))

        # one MMSI → many IMOs
        mmsi_to_imos: dict[int, set] = {}
        mmsi_to_idxs: dict[int, list] = {}
        for r in valid_recs:
            mmsi_to_imos.setdefault(r.mmsi, set()).add(r.imo)
            mmsi_to_idxs.setdefault(r.mmsi, []).append(r.raw_index)

        for mmsi, imos in mmsi_to_imos.items():
            if len(imos) > 1:
                self.conflicts.append(IdentityConflict(
                    conflict_type="one_mmsi_many_imo",
                    imo=None,
                    mmsi=mmsi,
                    record_indices=mmsi_to_idxs[mmsi],
                    description=f"MMSI {mmsi} maps to {len(imos)} IMOs: {imos}",
                ))

        # Exact duplicates (same IMO + MMSI)
        seen: dict[tuple, int] = {}
        for r in valid_recs:
            key = (r.imo, r.mmsi)
            if key in seen:
                self.conflicts.append(IdentityConflict(
                    conflict_type="duplicate",
                    imo=r.imo,
                    mmsi=r.mmsi,
                    record_indices=[seen[key], r.raw_index],
                    description=f"Duplicate IMO+MMSI pair ({r.imo}, {r.mmsi})",
                ))
            else:
                seen[key] = r.raw_index

        logger.info("Found %d conflicts.", len(self.conflicts))

    # ------------------------------------------------------------------
    # Step 4 – Resolve / deduplicate and assign stable vessel IDs
    # ------------------------------------------------------------------
    def resolve_identities(self) -> pd.DataFrame:
        logger.info("Resolving identities ...")

        df = self.raw.copy()
        df["_imo_class"] = df["imo"].apply(lambda x: classify_imo(int(x)))
        df["_mmsi_valid"] = df["mmsi"].apply(lambda x: is_valid_mmsi(int(x)))
        df["_issues"] = [", ".join(r.issues) if r.issues else "OK" for r in self.records]

        # Generate a stable vessel_id:
        #   - For valid IMO → use IMO-padded string
        #   - Otherwise → hash of MMSI + name
        def make_vessel_id(row):
            if row["_imo_class"] == "VALID":
                return f"IMO-{int(row['imo']):07d}"
            name_part = str(row.get("name", "")).strip().upper()[:20]
            h = hashlib.md5(f"{int(row['mmsi'])}-{name_part}".encode()).hexdigest()[:8]
            return f"MMSI-{h}"

        df["vessel_id"] = df.apply(make_vessel_id, axis=1)

        # Keep the most-recently-updated record per vessel_id
        df["_ts"] = pd.to_datetime(df["staticData_updateTimestamp"], errors="coerce")
        df_sorted = df.sort_values("_ts", ascending=False)
        resolved = df_sorted.drop_duplicates(subset=["vessel_id"], keep="first").copy()
        resolved.drop(columns=["_ts"], inplace=True)

        self.resolved_df = resolved
        logger.info("Resolved to %d unique vessels (from %d raw).", len(resolved), len(df))
        return resolved

    # ------------------------------------------------------------------
    # Step 5 – Change-tracking helper
    # ------------------------------------------------------------------
    def build_change_log(self) -> pd.DataFrame:
        """
        For each vessel_id that appears multiple times in the raw data,
        emit a row per attribute change (name, flag, mmsi).
        """
        if self.resolved_df is None:
            self.resolve_identities()

        df = self.raw.copy()
        df["_imo_class"] = df["imo"].apply(lambda x: classify_imo(int(x)))

        def make_vid(row):
            if row["_imo_class"] == "VALID":
                return f"IMO-{int(row['imo']):07d}"
            name_part = str(row.get("name", "")).strip().upper()[:20]
            h = hashlib.md5(f"{int(row['mmsi'])}-{name_part}".encode()).hexdigest()[:8]
            return f"MMSI-{h}"

        df["vessel_id"] = df.apply(make_vid, axis=1)
        df["_ts"] = pd.to_datetime(df["staticData_updateTimestamp"], errors="coerce")
        df = df.sort_values(["vessel_id", "_ts"])

        rows = []
        track_cols = ["name", "flag", "mmsi", "vessel_type"]
        for vid, grp in df.groupby("vessel_id"):
            if len(grp) < 2:
                continue
            prev = None
            for _, row in grp.iterrows():
                if prev is None:
                    prev = row
                    continue
                for col in track_cols:
                    old_val = str(prev[col])
                    new_val = str(row[col])
                    if old_val != new_val:
                        rows.append({
                            "vessel_id": vid,
                            "attribute": col,
                            "old_value": old_val,
                            "new_value": new_val,
                            "changed_at": row["_ts"],
                        })
                prev = row

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        total = len(self.records)
        issues_count = sum(1 for r in self.records if r.issues)
        return {
            "total_raw_records": total,
            "records_with_issues": issues_count,
            "valid_records": total - issues_count,
            "conflicts_detected": len(self.conflicts),
            "conflict_breakdown": {
                ct: sum(1 for c in self.conflicts if c.conflict_type == ct)
                for ct in ["one_imo_many_mmsi", "one_mmsi_many_imo", "duplicate"]
            },
            "unique_vessels_after_resolution": len(self.resolved_df) if self.resolved_df is not None else None,
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("../data/dataset.csv", low_memory=False)
    resolver = VesselIdentityResolver(df)
    resolver.parse_records()
    resolver.validate_identifiers()
    resolver.detect_conflicts()
    resolved = resolver.resolve_identities()
    change_log = resolver.build_change_log()

    print("\n=== Summary ===")
    for k, v in resolver.summary().items():
        print(f"  {k}: {v}")

    print("\n=== Sample Conflicts (first 5) ===")
    for c in resolver.conflicts[:5]:
        print(f"  [{c.conflict_type}] {c.description}")

    print("\n=== Sample Change Log ===")
    print(change_log.head(10).to_string(index=False))

    print("\n=== Resolved Sample (first 5 rows, key cols) ===")
    print(resolved[["vessel_id", "imo", "mmsi", "name", "flag", "_imo_class", "_issues"]].head().to_string(index=False))
