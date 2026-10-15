"""Collect direction-of-effect evidence from Open Targets for ReMedy's gate.

The collector maps ReMedy disease and UniProt protein IDs to Open Targets
identifiers, retrieves target-disease evidence, and retains only records with
both a target direction (gain/loss of function) and a trait direction
(risk/protective). It never assigns a direction where the source leaves it
unknown.

Output: data/processed/disease_target_effects.csv
Audit:  reports/evidence_gate/open_targets_id_mapping_audit.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
MAP_QUERY = """
query Map($terms: [String!]!, $entities: [String!]) {
  mapIds(queryTerms: $terms, entityNames: $entities) {
    mappings { term hits { id entity name } }
  }
}
"""
TARGET_QUERY = """
query Targets($ids: [String!]!) {
  targets(ensemblIds: $ids) {
    id
    proteinIds { id source }
  }
}
"""
EVIDENCE_QUERY = """
query DiseaseEvidence($efoId: String!, $ensemblIds: [String!]!, $cursor: String) {
  disease(efoId: $efoId) {
    evidences(ensemblIds: $ensemblIds, size: 3000, cursor: $cursor) {
      cursor
      rows {
        id
        datasourceId
        score
        directionOnTarget
        directionOnTrait
        target { id }
        disease { id }
      }
    }
  }
}
"""


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def gql(query: str, variables: dict) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json", "User-Agent": "ReMedy academic research pipeline"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload["data"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def batches(values: list[str], size: int = 250):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def select_disease_mapping(node_id: str, label: str, hits: list[dict]) -> dict | None:
    candidates = [hit for hit in hits if clean(hit.get("entity")).upper() == "DISEASE"]
    # An existing EFO ID is already canonical. Otherwise only take an exact
    # label result; fuzzy hits are preserved in the audit but are not accepted.
    if node_id.upper().startswith("EFO_"):
        return next((hit for hit in candidates if clean(hit.get("id")) == node_id), None)
    exact = [hit for hit in candidates if normalise(hit.get("name")) == normalise(label)]
    return exact[0] if len(exact) == 1 else None


def direction_to_effect(on_target: object, on_trait: object) -> str:
    target = clean(on_target).upper()
    trait = clean(on_trait).upper()
    target_sign = 1 if target in {"GOF", "GAIN_OF_FUNCTION"} else -1 if target in {"LOF", "LOSS_OF_FUNCTION"} else 0
    trait_sign = 1 if trait == "RISK" else -1 if trait == "PROTECTIVE" else 0
    sign = target_sign * trait_sign
    if sign == 1:
        return "INCREASED_ACTIVITY"
    if sign == -1:
        return "DECREASED_ACTIVITY"
    return ""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=root / "data" / "processed" / "nodes.parquet")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "disease_target_effects.csv")
    parser.add_argument("--audit-dir", type=Path, default=root / "reports" / "evidence_gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = pd.read_parquet(args.nodes)
    diseases = nodes.loc[nodes["node_type"].eq("disease"), ["node_id", "label"]].copy()
    proteins = nodes.loc[nodes["node_type"].eq("protein"), ["node_id"]].copy()
    diseases["node_id"] = diseases["node_id"].map(clean)
    proteins["node_id"] = proteins["node_id"].map(clean)
    proteins["uniprot_id"] = proteins["node_id"].str.replace(r"^UniProt:", "", regex=True)

    # Map disease node ID and label separately. Exact label matches protect
    # against accidentally selecting a merely similar disease.
    disease_terms = list(dict.fromkeys(diseases["node_id"].tolist() + diseases["label"].map(clean).tolist()))
    disease_maps: dict[str, list[dict]] = {}
    for group in batches(disease_terms):
        data = gql(MAP_QUERY, {"terms": group, "entities": ["disease"]})
        for entry in data["mapIds"]["mappings"]:
            disease_maps[clean(entry["term"])] = entry.get("hits", [])

    disease_to_efo: dict[str, str] = {}
    audit_rows = []
    for row in diseases.itertuples(index=False):
        hits = disease_maps.get(row.node_id, []) + disease_maps.get(clean(row.label), [])
        selected = select_disease_mapping(row.node_id, row.label, hits)
        disease_to_efo[row.node_id] = clean(selected.get("id")) if selected else ""
        audit_rows.append({
            "entity_type": "disease", "remedy_id": row.node_id, "label": row.label,
            "open_targets_id": disease_to_efo[row.node_id],
            "mapping_status": "MAPPED_EXACT" if selected else "UNRESOLVED",
            "candidate_hit_count": len(hits),
        })

    # Map UniProt accessions to Open Targets target IDs, then verify the
    # accession from Target.proteinIds before any evidence is accepted.
    protein_maps: dict[str, list[dict]] = {}
    for group in batches(proteins["uniprot_id"].tolist()):
        data = gql(MAP_QUERY, {"terms": group, "entities": ["target"]})
        for entry in data["mapIds"]["mappings"]:
            protein_maps[clean(entry["term"])] = entry.get("hits", [])
    proposed_target_ids = sorted({
        clean(hit.get("id")) for hits in protein_maps.values() for hit in hits
        if clean(hit.get("entity")).upper() == "TARGET" and clean(hit.get("id"))
    })
    target_proteins: dict[str, set[str]] = {}
    for group in batches(proposed_target_ids):
        data = gql(TARGET_QUERY, {"ids": group})
        for target in data.get("targets", []):
            if not target:
                continue
            target_proteins[clean(target["id"])] = {
                clean(item.get("id")) for item in target.get("proteinIds", [])
            }

    ensembl_to_protein: dict[str, str] = {}
    for row in proteins.itertuples(index=False):
        candidates = [
            clean(hit.get("id")) for hit in protein_maps.get(row.uniprot_id, [])
            if clean(hit.get("entity")).upper() == "TARGET"
            and row.uniprot_id in target_proteins.get(clean(hit.get("id")), set())
        ]
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            ensembl_to_protein[candidates[0]] = row.node_id
            status = "MAPPED_VERIFIED"
        else:
            status = "UNRESOLVED" if not candidates else "AMBIGUOUS"
        audit_rows.append({
            "entity_type": "protein", "remedy_id": row.node_id, "label": row.uniprot_id,
            "open_targets_id": candidates[0] if len(candidates) == 1 else "",
            "mapping_status": status, "candidate_hit_count": len(candidates),
        })

    evidence_rows = []
    ensembl_ids = sorted(ensembl_to_protein)
    for remedy_disease_id, efo_id in disease_to_efo.items():
        if not efo_id:
            continue
        cursor = None
        while True:
            data = gql(EVIDENCE_QUERY, {
                "efoId": efo_id, "ensemblIds": ensembl_ids, "cursor": cursor,
            })
            payload = (data.get("disease") or {}).get("evidences") or {"rows": [], "cursor": None}
            for evidence in payload["rows"]:
                effect = direction_to_effect(evidence.get("directionOnTarget"), evidence.get("directionOnTrait"))
                target_id = clean((evidence.get("target") or {}).get("id"))
                if not effect or target_id not in ensembl_to_protein:
                    continue
                evidence_rows.append({
                    "disease_id": remedy_disease_id,
                    "protein_id": ensembl_to_protein[target_id],
                    "disease_effect": effect,
                    "source": f"OpenTargets:{clean(evidence.get('datasourceId'))}",
                    "evidence_id": f"OpenTargets:{clean(evidence.get('id'))}",
                    "direction_on_target": clean(evidence.get("directionOnTarget")),
                    "direction_on_trait": clean(evidence.get("directionOnTrait")),
                    "evidence_score": evidence.get("score"),
                    "open_targets_disease_id": efo_id,
                    "open_targets_target_id": target_id,
                })
            cursor = payload.get("cursor")
            if not cursor:
                break

    columns = [
        "disease_id", "protein_id", "disease_effect", "source", "evidence_id",
        "direction_on_target", "direction_on_trait", "evidence_score",
        "open_targets_disease_id", "open_targets_target_id",
    ]
    output = pd.DataFrame(evidence_rows, columns=columns).drop_duplicates(
        ["disease_id", "protein_id", "evidence_id"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    pd.DataFrame(audit_rows).to_csv(args.audit_dir / "open_targets_id_mapping_audit.csv", index=False)
    print(f"Wrote {len(output):,} direction-of-effect records to {args.output}")
    print(f"Mapped {sum(bool(value) for value in disease_to_efo.values())}/{len(disease_to_efo)} diseases and {len(ensembl_to_protein)}/{len(proteins)} proteins.")


if __name__ == "__main__":
    main()
