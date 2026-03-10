"""
Vessel Dataset Exploration & EDA
==================================
Run this script to reproduce all analysis findings.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from vessel_identity import VesselIdentityResolver, classify_imo, is_valid_mmsi

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("../data/dataset.csv", low_memory=False)
print(f"Loaded {len(df)} records, {df.shape[1]} columns\n")

# ── 1. Column overview ─────────────────────────────────────────────────────
print("=== Columns & Null Counts ===")
null_pct = (df.isnull().sum() / len(df) * 100).round(1)
for col, pct in null_pct.items():
    print(f"  {col:<40} {pct:>5.1f}% null")

# ── 2. IMO Validation ──────────────────────────────────────────────────────
print("\n=== IMO Classification ===")
df["imo_class"] = df["imo"].apply(lambda x: classify_imo(int(x)))
print(df["imo_class"].value_counts().to_string())
print(f"\nIMO value_counts (top 10 duplicated):")
print(df["imo"].value_counts().head(10).to_string())

# ── 3. MMSI Validation ─────────────────────────────────────────────────────
print("\n=== MMSI Validity ===")
df["mmsi_valid"] = df["mmsi"].apply(lambda x: is_valid_mmsi(int(x)))
print(df["mmsi_valid"].value_counts().to_string())

# ── 4. Conflict detection summary ─────────────────────────────────────────
print("\n=== Conflict Summary ===")
valid_df = df[df["imo_class"] == "VALID"]
imo_mmsi_counts = valid_df.groupby("imo")["mmsi"].nunique()
print(f"  Valid-IMO vessels with 1 MMSI  : {(imo_mmsi_counts == 1).sum()}")
print(f"  Valid-IMO vessels with 2+ MMSI : {(imo_mmsi_counts > 1).sum()}")

mmsi_imo_counts = valid_df.groupby("mmsi")["imo"].nunique()
print(f"  Valid-MMSI with 1 IMO          : {(mmsi_imo_counts == 1).sum()}")
print(f"  Valid-MMSI with 2+ IMO         : {(mmsi_imo_counts > 1).sum()}")

# ── 5. Vessel type distribution ────────────────────────────────────────────
print("\n=== Vessel Types ===")
print(df["vessel_type"].value_counts().head(15).to_string())

# ── 6. Flag distribution ───────────────────────────────────────────────────
print("\n=== Top 15 Flags ===")
print(df["flag"].value_counts().head(15).to_string())

# ── 7. Built year distribution ─────────────────────────────────────────────
print("\n=== Built Year Distribution (valid-IMO only) ===")
yr = valid_df["builtYear"].dropna()
print(f"  Range: {int(yr.min())} – {int(yr.max())}")
print(f"  Median: {int(yr.median())}")
print(yr.value_counts().sort_index().tail(20).to_string())

# ── 8. Identity resolution summary ────────────────────────────────────────
print("\n=== Full Identity Resolution ===")
resolver = VesselIdentityResolver(df)
resolver.parse_records()
resolver.validate_identifiers()
resolver.detect_conflicts()
resolved = resolver.resolve_identities()
print(json_summary := resolver.summary())

change_log = resolver.build_change_log()
print(f"\n=== Change Log (first 10 rows) ===")
print(change_log.head(10).to_string(index=False) if not change_log.empty else "No changes detected.")

# ── 9. Sample ground-truth table ──────────────────────────────────────────
print("\n=== Sample Resolved Vessels (10 rows, key cols) ===")
key_cols = ["vessel_id", "imo", "mmsi", "name", "vessel_type", "flag",
            "builtYear", "_imo_class", "_issues"]
available = [c for c in key_cols if c in resolved.columns]
print(resolved[available].head(10).to_string(index=False))

print("\n✅ EDA complete.")
