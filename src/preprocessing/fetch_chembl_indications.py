"""Retrieve ChEMBL indication records for a bounded ReMedy candidate batch.

This script uses ChEMBL's drug_indication endpoint to identify exact or
name-matched known indications. It does not infer efficacy; it prevents a known
use from being shown as a novel hypothesis and sends ambiguous term matches to
review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


API = "https://www.ebi.ac.uk/chembl/api/data/drug_indication.json"


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def norm_term(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def norm_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).lower())


def get_records(drug_id: str, cache_dir: Path) -> tuple[list[dict], str]:
    key = hashlib.sha256(drug_id.encode("utf-8")).hexdigest()[:20]
    path = cache_dir / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached.get("records", []), clean(cached.get("error"))
    params = {"molecule_chembl_id": drug_id, "limit": 1000}
    try:
        with urlopen(f"{API}?{urlencode(params)}", timeout=45) as response:
            payload = json.load(response)
        records = payload.get("drug_indications", payload.get("results", []))
        response_data = {"records": records, "error": ""}
    except Exception as error:
        response_data = {"records": [], "error": str(error)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response_data), encoding="utf-8")
    time.sleep(0.12)
    return response_data["records"], clean(response_data["error"])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1" / "candidate_evidence_chain.csv",
    )
    parser.add_argument(
        "--disease-mapping", type=Path,
        default=root / "reports" / "evidence_gate" / "open_targets_id_mapping_audit.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "processed" / "chembl_indication_evidence.csv",
    )
    parser.add_argument(
        "--pair-status-output", type=Path,
        default=root / "reports" / "evidence_chain" / "chembl_known_use_screen.csv",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=root / "data" / "raw" / "chembl_indications",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(
        args.candidates, dtype={"drug_id": str, "disease_id": str}
    ).fillna("")
    required = {"candidate_id", "drug_id", "drug_name", "disease_id", "disease_name"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file missing columns: {sorted(missing)}")

    mapping = pd.read_csv(args.disease_mapping, dtype=str).fillna("")
    mapping = mapping.loc[mapping["entity_type"].eq("disease")].copy()
    efo_by_disease = mapping.set_index("remedy_id")["open_targets_id"].to_dict()

    unique_drugs = candidates[["drug_id", "drug_name"]].drop_duplicates()
    records = []
    retrieval_errors = {}
    for index, row in enumerate(unique_drugs.itertuples(index=False), start=1):
        if index == 1 or index % 10 == 0 or index == len(unique_drugs):
            print(f"ChEMBL indication retrieval {index}/{len(unique_drugs)}: {row.drug_name}", flush=True)
        indications, error = get_records(clean(row.drug_id), args.cache_dir)
        if error:
            retrieval_errors[clean(row.drug_id)] = error
        for indication in indications:
            records.append({
                "drug_id": clean(row.drug_id),
                "drug_name": clean(row.drug_name),
                "chembl_indication_id": clean(indication.get("drugind_id")),
                "indication_efo_id": clean(indication.get("efo_id")),
                "indication_efo_term": clean(indication.get("efo_term")),
                "indication_mesh_id": clean(indication.get("mesh_id")),
                "indication_mesh_heading": clean(indication.get("mesh_heading")),
                "max_phase_for_ind": clean(indication.get("max_phase_for_ind")),
                "source": "ChEMBL:drug_indication",
                "source_url": (
                    "https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/"
                    f"molecule_chembl_id:{clean(row.drug_id)}"
                ),
            })
    evidence = pd.DataFrame(records)
    if evidence.empty:
        evidence = pd.DataFrame(columns=[
            "drug_id", "drug_name", "chembl_indication_id", "indication_efo_id",
            "indication_efo_term", "indication_mesh_id", "indication_mesh_heading",
            "max_phase_for_ind", "source", "source_url",
        ])

    status_rows = []
    for candidate in candidates.itertuples(index=False):
        drug_records = evidence.loc[evidence["drug_id"].eq(clean(candidate.drug_id))]
        efo_id = clean(efo_by_disease.get(clean(candidate.disease_id), ""))
        exact = drug_records.loc[
            drug_records["indication_efo_id"].map(norm_id).eq(norm_id(efo_id))
        ] if efo_id else drug_records.iloc[0:0]
        term = drug_records.loc[
            drug_records["indication_efo_term"].map(norm_term).eq(norm_term(candidate.disease_name))
        ]
        if clean(candidate.drug_id) in retrieval_errors:
            state = "CHEMBL_RETRIEVAL_ERROR"
            matched = drug_records
        elif not exact.empty:
            state = "KNOWN_USE_EXACT_EFO_MATCH"
            matched = exact
        elif not term.empty:
            state = "KNOWN_USE_TERM_MATCH_REVIEW"
            matched = term
        elif drug_records.empty:
            state = "NO_CHEMBL_INDICATION_RECORD"
            matched = drug_records
        else:
            state = "NO_MATCHING_CHEMBL_INDICATION"
            matched = drug_records.iloc[0:0]
        status_rows.append({
            "candidate_id": candidate.candidate_id,
            "drug_id": candidate.drug_id,
            "drug_name": candidate.drug_name,
            "disease_id": candidate.disease_id,
            "disease_name": candidate.disease_name,
            "mapped_open_targets_disease_id": efo_id,
            "known_use_screen_status": state,
            "matched_indication_terms": " | ".join(
                matched["indication_efo_term"].dropna().astype(str).unique()
            ),
            "matched_indication_ids": " | ".join(
                matched["indication_efo_id"].dropna().astype(str).unique()
            ),
            "source": "ChEMBL:drug_indication",
            "retrieval_error": retrieval_errors.get(clean(candidate.drug_id), ""),
        })

    statuses = pd.DataFrame(status_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pair_status_output.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output, index=False)
    statuses.to_csv(args.pair_status_output, index=False)
    print(statuses["known_use_screen_status"].value_counts().to_string())
    print(f"Wrote {len(evidence):,} ChEMBL indication records to {args.output}")
    print(f"Wrote candidate known-use screen to {args.pair_status_output}")


if __name__ == "__main__":
    main()
