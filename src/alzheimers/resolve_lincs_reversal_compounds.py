"""Resolve LINCS reversal perturbations against ChEMBL and Alzheimer target context.

This script is intentionally conservative: it only accepts exact normalised ChEMBL
preferred-name matches.  It reports unresolved compounds rather than guessing
aliases, and keeps source-backed ChEMBL mechanism records separate from LINCS
cell-line reversal signals.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalised_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", clean(value).upper())


def joined(values: pd.Series) -> str:
    return " | ".join(sorted({clean(value) for value in values if clean(value)}))


def fetch_json(endpoint: str, params: dict[str, object], cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    query = urlencode(params, doseq=True, safe=",")
    request = Request(
        f"{CHEMBL_BASE}/{endpoint}.json?{query}",
        headers={"User-Agent": "ReMedy academic compound-resolution workflow"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def resolve_exact_chembl(perturbation: str, cache_dir: Path) -> dict:
    query_name = clean(perturbation)
    if not query_name:
        return {}
    cache = cache_dir / "search" / f"{normalised_name(query_name) or 'blank'}.json"
    payload = fetch_json("molecule/search", {"q": query_name, "limit": 20}, cache)
    hits = payload.get("molecules", [])
    exact = [
        hit for hit in hits
        if normalised_name(hit.get("pref_name")) == normalised_name(query_name)
    ]
    return exact[0] if len(exact) == 1 else {}


def mechanisms_for_molecule(molecule_id: str, cache_dir: Path) -> list[dict]:
    if not molecule_id:
        return []
    cache = cache_dir / "mechanisms" / f"{molecule_id}.json"
    payload = fetch_json(
        "mechanism",
        {"molecule_chembl_id": molecule_id, "limit": 1000, "offset": 0},
        cache,
    )
    return payload.get("mechanisms", [])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lincs_dir = args.curated_dir / "lincs_reversal"
    candidates_path = lincs_dir / "l1000_reversal_candidate_screen.csv"
    target_registry_path = args.curated_dir / "expanded_sources" / "chembl" / "target_registry.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing {candidates_path}. Run query_l1000_reversal_candidates.py first."
        )
    if not target_registry_path.exists():
        raise FileNotFoundError(f"Missing {target_registry_path}.")

    candidates = pd.read_csv(candidates_path, dtype=str).fillna("")
    candidates["best_l1000_reverse_rank"] = pd.to_numeric(
        candidates["best_l1000_reverse_rank"], errors="coerce"
    )
    candidates = candidates.sort_values("best_l1000_reverse_rank").head(args.limit).copy()
    target_registry = pd.read_csv(target_registry_path, dtype=str).fillna("")
    target_map = (
        target_registry.groupby("chembl_target_id", as_index=False)
        .agg(
            panel_protein_ids=("matched_protein_ids", joined),
            panel_protein_accessions=("matched_uniprot_accessions", joined),
            panel_target_names=("target_name", joined),
        )
    )

    cache_dir = args.curated_dir / "raw_cache" / "lincs_compound_resolution"
    resolution_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    for index, row in enumerate(candidates.itertuples(index=False), start=1):
        drug_name = clean(row.pert_desc)
        molecule = resolve_exact_chembl(drug_name, cache_dir)
        molecule_id = clean(molecule.get("molecule_chembl_id"))
        max_phase = molecule.get("max_phase", "")
        try:
            approved_or_late_stage = float(max_phase) >= 4
        except (TypeError, ValueError):
            approved_or_late_stage = False
        resolution_rows.append(
            {
                "normalized_drug_name": normalised_name(drug_name),
                "pert_desc": drug_name,
                "best_l1000_reverse_rank": row.best_l1000_reverse_rank,
                "chembl_molecule_id": molecule_id,
                "chembl_pref_name": clean(molecule.get("pref_name")),
                "chembl_max_phase": max_phase,
                "chembl_molecule_type": clean(molecule.get("molecule_type")),
                "exact_chembl_name_match": bool(molecule_id),
                "approved_or_phase4": approved_or_late_stage,
                "chembl_source_url": (
                    f"https://www.ebi.ac.uk/chembl/explore/compound/{molecule_id}"
                    if molecule_id else ""
                ),
            }
        )
        for mechanism in mechanisms_for_molecule(molecule_id, cache_dir):
            mechanism_rows.append(
                {
                    "chembl_molecule_id": molecule_id,
                    "chembl_target_id": clean(mechanism.get("target_chembl_id")),
                    "action_type": clean(mechanism.get("action_type")),
                    "mechanism_of_action": clean(mechanism.get("mechanism_of_action")),
                    "direct_interaction": bool(mechanism.get("direct_interaction")),
                    "chembl_mechanism_record_id": clean(mechanism.get("record_id")),
                }
            )
        if index % 10 == 0 or index == len(candidates):
            print(f"Resolved {index}/{len(candidates)} LINCS perturbations against ChEMBL.", flush=True)

    resolved = pd.DataFrame(resolution_rows)
    all_mechanisms = pd.DataFrame(mechanism_rows)
    mechanisms = all_mechanisms.copy()
    if mechanisms.empty:
        panel_context = pd.DataFrame(
            columns=[
                "chembl_molecule_id",
                "panel_protein_ids",
                "panel_protein_accessions",
                "panel_target_names",
                "panel_actions",
                "panel_mechanisms",
            ]
        )
    else:
        mechanisms = mechanisms.merge(target_map, on="chembl_target_id", how="inner")
        panel_context = (
            mechanisms.groupby("chembl_molecule_id", as_index=False)
            .agg(
                panel_protein_ids=("panel_protein_ids", joined),
                panel_protein_accessions=("panel_protein_accessions", joined),
                panel_target_names=("panel_target_names", joined),
                panel_actions=("action_type", joined),
                panel_mechanisms=("mechanism_of_action", joined),
            )
        )

    result = resolved.merge(panel_context, on="chembl_molecule_id", how="left")
    result["has_alzheimer_panel_mechanism"] = result["panel_protein_ids"].map(clean).ne("")
    result["candidate_resolution_status"] = result.apply(
        lambda row: (
            "LINCS_SIGNAL_UNRESOLVED_IN_CHEMBL"
            if not row["exact_chembl_name_match"]
            else "LINCS_SIGNAL_RESOLVED_NON_PHASE4"
            if not row["approved_or_phase4"]
            else "LINCS_SIGNAL_PHASE4_WITH_ALZHEIMER_PANEL_MECHANISM"
            if row["has_alzheimer_panel_mechanism"]
            else "LINCS_SIGNAL_PHASE4_WITHOUT_ALZHEIMER_PANEL_MECHANISM"
        ),
        axis=1,
    )

    output_dir = lincs_dir / "compound_resolution"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "lincs_reversal_chembl_resolved_candidates.csv", index=False)
    all_mechanisms.to_csv(output_dir / "lincs_reversal_all_chembl_mechanisms.csv", index=False)
    mechanisms.to_csv(output_dir / "lincs_reversal_chembl_mechanisms.csv", index=False)
    print(
        json.dumps(
            {
                "lincs_perturbations_screened": len(result),
                "exact_chembl_resolutions": int(result["exact_chembl_name_match"].sum()),
                "phase4_or_approved_resolutions": int(result["approved_or_phase4"].sum()),
                "phase4_with_alzheimer_panel_mechanism": int(
                    result["candidate_resolution_status"].eq(
                        "LINCS_SIGNAL_PHASE4_WITH_ALZHEIMER_PANEL_MECHANISM"
                    ).sum()
                ),
                "output": str(output_dir / "lincs_reversal_chembl_resolved_candidates.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
