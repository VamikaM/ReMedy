"""Expand ReMedy's Alzheimer V1 drug catalogue from ChEMBL.

The broad ReMedy graph contains mechanisms for only its original drug universe.
This collector starts with the Alzheimer-linked protein panel and retrieves
source-backed ChEMBL mechanisms for phase-4 (marketed) medicines targeting
those proteins. It builds an evidence catalogue, not a therapeutic shortlist.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1_000


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def fetch_json(endpoint: str, params: dict[str, object], cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    query = urlencode({**params, "limit": PAGE_SIZE, "offset": 0}, doseq=True, safe=",")
    request = Request(
        f"{BASE_URL}/{endpoint}.json?{query}",
        headers={"User-Agent": "ReMedy academic evidence catalogue"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def paged(
    endpoint: str, key: str, params: dict[str, object], cache_file: Path
) -> list[dict]:
    if cache_file.exists():
        return [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines() if line]
    output: list[dict] = []
    offset = 0
    while True:
        query = urlencode({**params, "limit": PAGE_SIZE, "offset": offset}, doseq=True, safe=",")
        request = Request(
            f"{BASE_URL}/{endpoint}.json?{query}",
            headers={"User-Agent": "ReMedy academic evidence catalogue"},
        )
        for attempt in range(4):
            try:
                with urlopen(request, timeout=90) as response:
                    payload = json.load(response)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        output.extend(payload.get(key, []))
        if not payload.get("page_meta", {}).get("next"):
            break
        offset += PAGE_SIZE
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        "\n".join(json.dumps(row) for row in output) + ("\n" if output else ""),
        encoding="utf-8",
    )
    return output


def chunked(values: list[str], size: int = 75):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-profile",
        type=Path,
        default=curated / "entities" / "disease_target_profile.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=curated / "expanded_sources" / "chembl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "alzheimers_chembl_catalog",
    )
    return parser.parse_args()


def accession(protein_id: str) -> str:
    return clean(protein_id).split(":")[-1].upper()


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.target_profile, dtype=str).fillna("")
    required = {"protein_id", "protein_label"}
    if missing := required - set(targets.columns):
        raise ValueError(f"target profile missing columns: {sorted(missing)}")
    targets["protein_id"] = targets["protein_id"].map(clean)
    targets["accession"] = targets["protein_id"].map(accession)
    target_by_accession = dict(zip(targets["accession"], targets["protein_id"]))
    label_by_protein = dict(zip(targets["protein_id"], targets["protein_label"]))

    # Map every Alzheimer-panel UniProt accession to its ChEMBL target records.
    chembl_target_rows: list[dict] = []
    target_to_accessions: dict[str, set[str]] = {}
    for index, protein_accession in enumerate(sorted(target_by_accession), start=1):
        records = paged(
            "target",
            "targets",
            {"target_components__accession": protein_accession},
            args.cache_dir / "targets" / f"{protein_accession}.jsonl",
        )
        for record in records:
            target_id = clean(record.get("target_chembl_id"))
            if not target_id:
                continue
            component_accessions = {
                clean(component.get("accession")).upper()
                for component in record.get("target_components", [])
                if clean(component.get("accession"))
            }
            matched = sorted(component_accessions & set(target_by_accession))
            if not matched:
                continue
            target_to_accessions.setdefault(target_id, set()).update(matched)
            chembl_target_rows.append(
                {
                    "chembl_target_id": target_id,
                    "target_type": clean(record.get("target_type")),
                    "target_name": clean(record.get("pref_name")),
                    "matched_uniprot_accessions": " | ".join(matched),
                    "matched_protein_ids": " | ".join(target_by_accession[item] for item in matched),
                }
            )
        if index % 20 == 0 or index == len(target_by_accession):
            print(f"Mapped ChEMBL targets for {index}/{len(target_by_accession)} Alzheimer proteins.", flush=True)

    target_registry = pd.DataFrame(chembl_target_rows).drop_duplicates()
    target_registry = target_registry.sort_values(["chembl_target_id", "target_name"])
    target_registry.to_csv(args.output_dir / "target_registry.csv", index=False) if args.output_dir.exists() else None

    # Retrieve mechanism records by target, then filter to phase-4 molecules.
    mechanisms: list[dict] = []
    chembl_target_ids = sorted(target_to_accessions)
    for index, chembl_target_id in enumerate(chembl_target_ids, start=1):
        records = paged(
            "mechanism",
            "mechanisms",
            {"target_chembl_id": chembl_target_id},
            args.cache_dir / "mechanisms" / f"{chembl_target_id}.jsonl",
        )
        for record in records:
            record["_chembl_target_id"] = chembl_target_id
            mechanisms.append(record)
        if index % 20 == 0 or index == len(chembl_target_ids):
            print(f"Retrieved mechanisms for {index}/{len(chembl_target_ids)} ChEMBL targets.", flush=True)

    molecule_ids = sorted({
        clean(record.get("molecule_chembl_id")) for record in mechanisms
        if clean(record.get("molecule_chembl_id"))
    })
    molecule_rows: list[dict] = []
    for index, group in enumerate(chunked(molecule_ids), start=1):
        records = paged(
            "molecule",
            "molecules",
            {"molecule_chembl_id__in": ",".join(group)},
            args.cache_dir / "molecules" / f"batch_{index:04d}.jsonl",
        )
        molecule_rows.extend(records)
        print(f"Retrieved molecule metadata batch {index}.", flush=True)

    molecules = pd.DataFrame(molecule_rows)
    if molecules.empty:
        raise RuntimeError("No ChEMBL molecule metadata was retrieved for Alzheimer target mechanisms.")
    molecules["molecule_chembl_id"] = molecules["molecule_chembl_id"].map(clean)
    molecules["max_phase"] = pd.to_numeric(molecules.get("max_phase"), errors="coerce").fillna(0)
    approved = molecules.loc[molecules["max_phase"].ge(4)].copy()
    approved_ids = set(approved["molecule_chembl_id"])

    drug_rows: list[dict] = []
    for mechanism in mechanisms:
        drug_id = clean(mechanism.get("molecule_chembl_id"))
        if drug_id not in approved_ids:
            continue
        chembl_target_id = clean(mechanism.get("_chembl_target_id"))
        for target_accession in sorted(target_to_accessions.get(chembl_target_id, set())):
            protein_id = target_by_accession[target_accession]
            drug_rows.append(
                {
                    "drug_id": drug_id,
                    "protein_id": protein_id,
                    "protein_label": label_by_protein.get(protein_id, ""),
                    "chembl_target_id": chembl_target_id,
                    "action_type": clean(mechanism.get("action_type")) or "UNKNOWN",
                    "mechanism_of_action": clean(mechanism.get("mechanism_of_action")),
                    "direct_interaction": bool(mechanism.get("direct_interaction")),
                    "source": "ChEMBL",
                    "evidence_id": f"CHEMBL-MECH:{clean(mechanism.get('record_id'))}",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_registry.to_csv(args.output_dir / "target_registry.csv", index=False)
    molecule_columns = [
        "molecule_chembl_id", "pref_name", "max_phase", "molecule_type",
        "first_approval", "oral", "black_box_warning",
    ]
    available_molecule_columns = [column for column in molecule_columns if column in approved.columns]
    approved[available_molecule_columns].drop_duplicates().sort_values(
        ["pref_name", "molecule_chembl_id"], na_position="last"
    ).to_csv(args.output_dir / "approved_molecule_registry.csv", index=False)

    catalogue_columns = [
        "drug_id", "protein_id", "protein_label", "chembl_target_id", "action_type",
        "mechanism_of_action", "direct_interaction", "source", "evidence_id",
    ]
    catalogue = pd.DataFrame(drug_rows, columns=catalogue_columns).drop_duplicates()
    catalogue = catalogue.sort_values(["protein_label", "drug_id", "action_type"])
    catalogue.to_csv(args.output_dir / "approved_drug_target_catalog.csv", index=False)

    summary = {
        "alzheimers_panel_proteins": int(len(targets)),
        "panel_proteins_mapped_to_chembl_targets": int(
            len({protein for accessions in target_to_accessions.values() for protein in accessions})
        ),
        "chembl_targets_found": int(len(chembl_target_ids)),
        "mechanism_records_retrieved": int(len(mechanisms)),
        "phase_4_molecules": int(len(approved)),
        "source_backed_phase_4_drug_target_rows": int(len(catalogue)),
        "unique_phase_4_drugs_with_panel_mechanisms": int(catalogue["drug_id"].nunique()),
        "policy": (
            "This is a ChEMBL source catalogue. Target overlap and recorded action do not "
            "establish Alzheimer disease benefit; disease relevance, exposure, outcome, and "
            "safety evidence are evaluated in subsequent stages."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote expanded ChEMBL Alzheimer catalogue to {args.output_dir}")


if __name__ == "__main__":
    main()
