"""Collect source-backed drug action records for ReMedy's evidence gate.

The collector downloads and caches ChEMBL mechanisms plus target-component mappings,
then retains only actions that map to protein identifiers already present in the
ReMedy graph.  It does not infer an action from target overlap.

Output: data/processed/drug_mechanisms.csv
Required columns for evidence_gate.py:
drug_id, protein_id, action_type, source, evidence_id
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1_000


def fetch_json(endpoint: str, params: dict[str, object]) -> dict:
    url = f"{BASE_URL}/{endpoint}.json?{urlencode(params, doseq=True, safe=',')}"
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def paged(endpoint: str, record_key: str, params: dict[str, object]) -> list[dict]:
    records: list[dict] = []
    offset = 0
    while True:
        page = fetch_json(endpoint, {**params, "limit": PAGE_SIZE, "offset": offset})
        records.extend(page.get(record_key, []))
        if not page.get("page_meta", {}).get("next"):
            return records
        offset += PAGE_SIZE


def chunks(values: list[str], size: int = 75):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_or_fetch(path: Path, endpoint: str, record_key: str,
                  ids: list[str], field: str) -> list[dict]:
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    results: list[dict] = []
    for index, group in enumerate(chunks(ids), start=1):
        results.extend(paged(endpoint, record_key, {f"{field}__in": ",".join(group)}))
        print(f"{endpoint}: completed batch {index}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=root / "data" / "processed" / "nodes.parquet")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "drug_mechanisms.csv")
    parser.add_argument("--cache-dir", type=Path, default=root / "data" / "raw" / "chembl_evidence_gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = pd.read_parquet(args.nodes)
    drugs = sorted(nodes.loc[nodes["node_type"].eq("drug"), "node_id"].astype(str))
    graph_proteins = nodes.loc[nodes["node_type"].eq("protein"), "node_id"].astype(str)

    # Accept the graph's canonical protein ID exactly as stored, while matching
    # ChEMBL target-component accessions independently of prefix/case.
    def accession(node_id: str) -> str:
        return str(node_id).split(":")[-1].upper()

    protein_by_accession = {
        accession(node_id): node_id for node_id in graph_proteins
        if accession(node_id)
    }

    mechanisms = load_or_fetch(
        args.cache_dir / "mechanisms.jsonl", "mechanism", "mechanisms", drugs,
        "molecule_chembl_id",
    )
    target_ids = sorted({row.get("target_chembl_id") for row in mechanisms if row.get("target_chembl_id")})
    targets = load_or_fetch(
        args.cache_dir / "targets.jsonl", "target", "targets", target_ids,
        "target_chembl_id",
    )
    target_to_accessions: dict[str, set[str]] = {}
    for target in targets:
        accessions = {
            component.get("accession")
            for component in target.get("target_components", [])
            if component.get("component_type") == "PROTEIN" and component.get("accession")
        }
        target_to_accessions[str(target.get("target_chembl_id"))] = accessions

    rows = []
    for mechanism in mechanisms:
        drug_id = str(mechanism.get("molecule_chembl_id") or mechanism.get("parent_molecule_chembl_id") or "")
        if drug_id not in drugs:
            continue
        for accession in target_to_accessions.get(str(mechanism.get("target_chembl_id")), set()):
            protein_id = protein_by_accession.get(str(accession).upper())
            if not protein_id:
                continue
            rows.append({
                "drug_id": drug_id,
                "protein_id": protein_id,
                "action_type": mechanism.get("action_type") or "UNKNOWN",
                "mechanism_of_action": mechanism.get("mechanism_of_action") or "",
                "direct_interaction": int(bool(mechanism.get("direct_interaction"))),
                "source": "ChEMBL",
                "evidence_id": f"CHEMBL-MECH:{mechanism.get('record_id')}",
            })

    columns = [
        "drug_id", "protein_id", "action_type", "mechanism_of_action",
        "direct_interaction", "source", "evidence_id",
    ]
    output = pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["drug_id", "protein_id", "action_type", "evidence_id"]
    ).sort_values(["drug_id", "protein_id", "action_type"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} source-backed action records to {args.output}")
    print(f"Graph proteins considered: {len(graph_proteins):,}; matched ChEMBL accessions: {output['protein_id'].nunique():,}")
    print("Action type counts:")
    print(output["action_type"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
