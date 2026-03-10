import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate


DATA_PATH = "docs/dataset.csv"


PLACEHOLDER_IMOS = {0, 1000000, 8000000}


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def is_valid_imo(imo: Any) -> bool:
    try:
        v = int(imo)
    except (TypeError, ValueError):
        return False
    if v in PLACEHOLDER_IMOS:
        return False
    if v <= 0:
        return False
    return True


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names (lowercase for convenience)
    df.columns = [c.strip() for c in df.columns]

    # Add helper flags
    df["imo_valid"] = df["imo"].apply(is_valid_imo)

    # Normalize some text fields
    for col in ["name", "vessel_type", "flag", "destination"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": ""})

    return df


def dataset_summary(df: pd.DataFrame) -> None:
    total = len(df)
    valid_imo = int(df["imo"].apply(is_valid_imo).sum())
    unique_imo = df.loc[df["imo"].apply(is_valid_imo), "imo"].nunique()
    unique_mmsi = df["mmsi"].nunique()

    print("== Dataset Summary ==")
    print(f"Rows: {total}")
    print(f"Valid IMO rows: {valid_imo}")
    print(f"Unique valid IMOs: {unique_imo}")
    print(f"Unique MMSIs: {unique_mmsi}")
    print()

    # Show top vessel types and flags
    for col, title in [("vessel_type", "Top Vessel Types"), ("flag", "Top Flags")]:
        if col in df.columns:
            print(f"== {title} ==")
            counts = df[col].value_counts().head(10)
            print(tabulate(counts.reset_index().values, headers=[col, "count"]))
            print()


def build_identity_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple heuristic identity clustering:
    - Primary key: valid IMO.
    - Fallback: (mmsi, lower(name)) where name is non-empty.
    - Fallback2: (lower(name), flag) for records without IMO and MMSI.

    The result adds:
    - vessel_id: cluster identifier.
    - cluster_size: size of each cluster.
    - cluster_conflict_type: flag for simple conflicts.
    """
    df = df.copy()

    # Canonical helper columns
    df["name_norm"] = df["name"].astype(str).str.lower().str.strip()
    df["flag_norm"] = df["flag"].astype(str).str.upper().str.strip()

    # Start with a default cluster key as index
    cluster_key = pd.Series(df.index.astype(str), index=df.index, dtype="object")

    # 1) Use valid IMO where available
    valid_imo_mask = df["imo"].apply(is_valid_imo)
    cluster_key.loc[valid_imo_mask] = "IMO_" + df.loc[valid_imo_mask, "imo"].astype(int).astype(str)

    # 2) Use MMSI+name_norm where IMO invalid but MMSI present and name non-empty
    no_valid_imo = ~valid_imo_mask
    has_mmsi = df["mmsi"].notna()
    has_name = df["name_norm"] != ""
    mask_mmsi_name = no_valid_imo & has_mmsi & has_name
    cluster_key.loc[mask_mmsi_name] = (
        "MMSI_NAME_"
        + df.loc[mask_mmsi_name, "mmsi"].astype(int).astype(str)
        + "_"
        + df.loc[mask_mmsi_name, "name_norm"]
    )

    # 3) Use name+flag for rows with neither valid IMO nor MMSI but with a name
    no_mmsi = df["mmsi"].isna()
    mask_name_flag = no_valid_imo & no_mmsi & has_name
    cluster_key.loc[mask_name_flag] = (
        "NAME_FLAG_"
        + df.loc[mask_name_flag, "name_norm"]
        + "_"
        + df.loc[mask_name_flag, "flag_norm"]
    )

    df["cluster_key"] = cluster_key

    # Derive vessel_id as a compact identifier
    # Use numeric part of IMO when possible, else a hash of the cluster key.
    vessel_ids: Dict[str, str] = {}
    for key in df["cluster_key"].unique():
        if key.startswith("IMO_"):
            try:
                imo_val = int(key.split("_", 1)[1])
                vessel_ids[key] = f"V{imo_val}"
            except Exception:
                vessel_ids[key] = f"VHASH_{abs(hash(key)) % 10_000_000}"
        else:
            vessel_ids[key] = f"VHASH_{abs(hash(key)) % 10_000_000}"

    df["vessel_id"] = df["cluster_key"].map(vessel_ids)

    # Compute cluster-level stats to detect conflicts
    agg = (
        df.groupby("vessel_id")
        .agg(
            rows=("imo", "size"),
            distinct_imo=("imo", lambda s: set(int(x) for x in s.dropna())),
            distinct_mmsi=("mmsi", lambda s: set(int(x) for x in s.dropna())),
            distinct_name=("name_norm", lambda s: set(x for x in s.dropna() if x)),
            distinct_flag=("flag_norm", lambda s: set(x for x in s.dropna() if x)),
        )
        .reset_index()
    )

    def conflict_type(row: pd.Series) -> str:
        imos = {i for i in row["distinct_imo"] if is_valid_imo(i)}
        mmsis = row["distinct_mmsi"]
        names = row["distinct_name"]
        flags = row["distinct_flag"]

        if len(imos) > 1:
            return "IMO_CONFLICT"
        if len(mmsis) > 5:
            return "MMSI_MANY"
        if len(names) > 3 or len(flags) > 3:
            return "ATTRIBUTE_DRIFT"
        return "CLEAN"

    agg["cluster_conflict_type"] = agg.apply(conflict_type, axis=1)

    df = df.merge(
        agg[["vessel_id", "rows", "cluster_conflict_type"]],
        on="vessel_id",
        how="left",
    )
    df = df.rename(columns={"rows": "cluster_size"})

    return df


def print_cluster_examples(df_clusters: pd.DataFrame, limit: int = 10) -> None:
    # Show some clusters with conflicts, then some clean ones
    conflict = df_clusters[df_clusters["cluster_conflict_type"] != "CLEAN"]
    clean = df_clusters[df_clusters["cluster_conflict_type"] == "CLEAN"]

    print("== Example identity clusters with conflicts ==")
    if conflict.empty:
        print("(none found)")
    else:
        for vessel_id, group in list(conflict.groupby("vessel_id"))[: limit // 2]:
            print(f"\nVessel ID: {vessel_id} (size={group['cluster_size'].iloc[0]})")
            print(f"Conflict type: {group['cluster_conflict_type'].iloc[0]}")
            cols = ["imo", "mmsi", "name", "flag", "vessel_type"]
            present_cols = [c for c in cols if c in group.columns]
            print(tabulate(group[present_cols].head(5).values, headers=present_cols))

    print("\n== Example clean identity clusters ==")
    if clean.empty:
        print("(none found)")
    else:
        for vessel_id, group in list(clean.groupby("vessel_id"))[: limit // 2]:
            print(f"\nVessel ID: {vessel_id} (size={group['cluster_size'].iloc[0]})")
            print(f"Conflict type: {group['cluster_conflict_type'].iloc[0]}")
            cols = ["imo", "mmsi", "name", "flag", "vessel_type"]
            present_cols = [c for c in cols if c in group.columns]
            print(tabulate(group[present_cols].head(5).values, headers=present_cols))


def search_vessels(
    df: pd.DataFrame,
    imo: Optional[int] = None,
    mmsi: Optional[int] = None,
    name_contains: Optional[str] = None,
    vessel_type: Optional[str] = None,
    flag: Optional[str] = None,
    min_deadweight: Optional[float] = None,
    max_deadweight: Optional[float] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if imo is not None:
        mask &= df["imo"] == imo
    if mmsi is not None:
        mask &= df["mmsi"] == mmsi
    if name_contains:
        name_lower = df["name"].astype(str).str.lower()
        mask &= name_lower.str.contains(name_contains.lower())
    if vessel_type:
        vt_lower = df["vessel_type"].astype(str).str.lower()
        mask &= vt_lower == vessel_type.lower()
    if flag:
        flag_upper = df["flag"].astype(str).str.upper()
        mask &= flag_upper == flag.upper()
    if "deadweight" in df.columns:
        if min_deadweight is not None:
            mask &= df["deadweight"].fillna(0) >= min_deadweight
        if max_deadweight is not None:
            mask &= df["deadweight"].fillna(0) <= max_deadweight

    return df[mask]


def print_search_results(df: pd.DataFrame, limit: int = 20) -> None:
    if df.empty:
        print("No matching vessels found.")
        return

    cols = [
        "imo",
        "mmsi",
        "name",
        "vessel_type",
        "flag",
        "deadweight",
        "grossTonnage",
        "destination",
    ]
    present_cols = [c for c in cols if c in df.columns]
    out = df[present_cols].head(limit)
    print(tabulate(out.values, headers=present_cols))
    if len(df) > limit:
        print(f"... {len(df) - limit} more rows not shown ...")


@dataclass
class SessionState:
    last_filters: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[Tuple[Tuple[str, Any], ...], pd.DataFrame] = field(default_factory=dict)


def normalized_filters_key(filters: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(filters.items()))


def run_structured_search_with_cache(
    df: pd.DataFrame, session: SessionState, filters: Dict[str, Any]
) -> pd.DataFrame:
    key = normalized_filters_key(filters)
    if key in session.cache:
        return session.cache[key]

    result = search_vessels(
        df,
        imo=filters.get("imo"),
        mmsi=filters.get("mmsi"),
        name_contains=filters.get("name_contains"),
        vessel_type=filters.get("vessel_type"),
        flag=filters.get("flag"),
        min_deadweight=filters.get("min_deadweight"),
        max_deadweight=filters.get("max_deadweight"),
    )
    session.cache[key] = result
    session.last_filters = filters
    return result


def parse_chat_query_to_filters(
    text: str, previous_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Very simple parser for queries like:
      "show tankers flag=SG deadweight>30000"
      "find vessels name~HOEGH flag=LR"
    """
    tokens = text.strip().split()
    filters: Dict[str, Any] = dict(previous_filters or {})

    for tok in tokens:
        if "=" in tok or ">" in tok or "<" in tok or "~" in tok:
            # handle name~substring
            if "~" in tok:
                key, value = tok.split("~", 1)
                key = key.lower()
                if key == "name":
                    filters["name_contains"] = value
                continue

            # handle comparisons for deadweight
            op = None
            if ">" in tok:
                key, value = tok.split(">", 1)
                op = ">"
            elif "<" in tok:
                key, value = tok.split("<", 1)
                op = "<"
            else:
                key, value = tok.split("=", 1)

            key = key.lower()
            value = value.strip()

            if key in {"imo", "mmsi"}:
                try:
                    num = int(value)
                except ValueError:
                    continue
                filters[key] = num
            elif key in {"flag", "vessel_type"}:
                filters[key] = value
            elif key in {"deadweight"}:
                try:
                    num = float(value)
                except ValueError:
                    continue
                if op == ">":
                    filters["min_deadweight"] = num
                elif op == "<":
                    filters["max_deadweight"] = num
            # unknown keys are ignored

    # Special handling for "tankers" in free text -> vessel_type filter
    text_lower = text.lower()
    if "tankers" in text_lower or "tanker" in text_lower:
        # Only set if user hasn't provided a more specific vessel_type
        filters.setdefault("vessel_type", "Chemical Tanker")

    return filters


def chat_repl(df: pd.DataFrame) -> None:
    print("Conversational vessel search demo.")
    print("Type queries like: 'show tankers flag=SG deadweight>30000 name~HOEGH'")
    print("Commands: 'repeat last', 'clear cache', 'exit'")
    session = SessionState()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if line.lower().startswith("repeat last"):
            if not session.last_filters:
                print("No previous filters to repeat.")
                continue
            filters = session.last_filters
        elif line.lower().startswith("clear cache"):
            session.cache.clear()
            print("Cache cleared.")
            continue
        else:
            filters = parse_chat_query_to_filters(line, previous_filters=None)

        if not filters:
            print("Could not parse any filters from your input.")
            continue

        results = run_structured_search_with_cache(df, session, filters)
        print_search_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vessel identification and search POC."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary command
    subparsers.add_parser("summary", help="Show basic dataset summary.")

    # resolve command
    subparsers.add_parser(
        "resolve", help="Run simple identity resolution and show example clusters."
    )

    # search command
    search_parser = subparsers.add_parser(
        "search", help="Run a structured search over the vessel dataset."
    )
    search_parser.add_argument("--imo", type=int)
    search_parser.add_argument("--mmsi", type=int)
    search_parser.add_argument("--name-contains", type=str)
    search_parser.add_argument("--vessel-type", type=str)
    search_parser.add_argument("--flag", type=str)
    search_parser.add_argument("--min-deadweight", type=float)
    search_parser.add_argument("--max-deadweight", type=float)

    # chat command
    subparsers.add_parser(
        "chat",
        help="Run a simple conversational shell that translates text queries into structured searches with caching.",
    )

    args = parser.parse_args()

    df = load_dataset(DATA_PATH)
    df = clean_dataset(df)

    if args.command == "summary":
        dataset_summary(df)
    elif args.command == "resolve":
        df_clusters = build_identity_clusters(df)
        print_cluster_examples(df_clusters)
    elif args.command == "search":
        result = search_vessels(
            df,
            imo=args.imo,
            mmsi=args.mmsi,
            name_contains=args.name_contains,
            vessel_type=args.vessel_type,
            flag=args.flag,
            min_deadweight=args.min_deadweight,
            max_deadweight=args.max_deadweight,
        )
        print_search_results(result)
    elif args.command == "chat":
        chat_repl(df)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

