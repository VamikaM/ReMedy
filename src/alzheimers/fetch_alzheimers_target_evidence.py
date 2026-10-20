"""Collect and summarize Open Targets evidence for every Alzheimer V1 panel protein.

The output ranks *evidence coverage* for targets, not therapeutic efficacy. It
keeps all source evidence, including rows without directional genetics, so a
target is not artificially promoted just because a sign was available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
MAP = """query Map($terms: [String!]!, $entities: [String!]) { mapIds(queryTerms: $terms, entityNames: $entities) { mappings { term hits { id entity name } } } }"""
TARGETS = """query Targets($ids: [String!]!) { targets(ensemblIds: $ids) { id proteinIds { id source } } }"""
EVIDENCE = """query Evidence($efoId: String!, $ids: [String!]!, $cursor: String) { disease(efoId: $efoId) { evidences(ensemblIds: $ids, size: 3000, cursor: $cursor) { cursor rows { id datasourceId score directionOnTarget directionOnTrait target { id approvedSymbol approvedName } } } } }"""


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def chunks(values: list[str], size: int = 75):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def request(query: str, variables: dict, cache_dir: Path) -> dict:
    key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "ReMedy academic evidence catalogue"},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urlopen(req, timeout=90) as response:
                result = json.load(response)
            if result.get("errors"):
                raise RuntimeError(result["errors"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result["data"]), encoding="utf-8")
            return result["data"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def map_disease(disease_id: str, disease_name: str, hits: list[dict]) -> str:
    candidates = [
        item for item in hits
        if clean(item.get("entity")).upper() == "DISEASE" and clean(item.get("id"))
    ]
    expected = disease_id.replace(":", "_")
    exact_id = [item for item in candidates if clean(item.get("id")) == expected]
    if len(exact_id) == 1:
        return clean(exact_id[0]["id"])
    exact_name = [item for item in candidates if normalise(item.get("name")) == normalise(disease_name)]
    return clean(exact_name[0]["id"]) if len(exact_name) == 1 else ""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=curated / "expanded_sources" / "open_targets",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "alzheimers_open_targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    disease = pd.read_csv(args.curated_dir / "entities" / "disease.csv", dtype=str).fillna("")
    targets = pd.read_csv(
        args.curated_dir / "entities" / "disease_target_profile.csv", dtype=str
    ).fillna("")
    if len(disease) != 1:
        raise ValueError("Expected exactly one disease row in the Alzheimer V1 dataset.")
    disease_id = clean(disease.iloc[0]["disease_id"])
    disease_name = clean(disease.iloc[0]["disease_name"])
    targets["protein_id"] = targets["protein_id"].map(clean)
    targets["accession"] = targets["protein_id"].map(lambda value: value.split(":")[-1].upper())

    # Map the disease and UniProt accessions to verified Open Targets Ensembl IDs.
    terms = [disease_id, disease_name]
    data = request(MAP, {"terms": terms, "entities": ["disease"]}, args.cache_dir / "mapping")
    hit_map = {clean(row["term"]): row.get("hits", []) for row in data["mapIds"]["mappings"]}
    ot_disease_id = map_disease(
        disease_id, disease_name, hit_map.get(disease_id, []) + hit_map.get(disease_name, [])
    )
    if not ot_disease_id:
        raise RuntimeError(f"Could not map {disease_name} ({disease_id}) to an Open Targets disease ID.")

    target_hit_map: dict[str, list[dict]] = {}
    for group in chunks(sorted(targets["accession"].unique().tolist())):
        data = request(MAP, {"terms": group, "entities": ["target"]}, args.cache_dir / "mapping")
        target_hit_map.update({
            clean(row["term"]).upper(): row.get("hits", [])
            for row in data["mapIds"]["mappings"]
        })

    possible_ensembl = sorted({
        clean(hit.get("id"))
        for hits in target_hit_map.values()
        for hit in hits
        if clean(hit.get("entity")).upper() == "TARGET" and clean(hit.get("id"))
    })
    ensembl_to_accessions: dict[str, set[str]] = {}
    for group in chunks(possible_ensembl):
        data = request(TARGETS, {"ids": group}, args.cache_dir / "mapping")
        for item in data.get("targets", []):
            if item:
                ensembl_to_accessions[clean(item["id"])] = {
                    clean(record.get("id")).upper()
                    for record in item.get("proteinIds", [])
                    if clean(record.get("id"))
                }

    accession_to_protein = dict(zip(targets["accession"], targets["protein_id"]))
    protein_to_ensembl: dict[str, str] = {}
    for accession, protein_id in accession_to_protein.items():
        exact = sorted({
            clean(hit.get("id"))
            for hit in target_hit_map.get(accession, [])
            if clean(hit.get("entity")).upper() == "TARGET"
            and accession in ensembl_to_accessions.get(clean(hit.get("id")), set())
        })
        if len(exact) == 1:
            protein_to_ensembl[protein_id] = exact[0]

    # Retrieve every evidence row; do not drop directionless or non-genetic records.
    raw_rows: list[dict] = []
    inverse = {ensembl: protein for protein, ensembl in protein_to_ensembl.items()}
    for group_index, group in enumerate(chunks(sorted(inverse)), start=1):
        cursor = None
        while True:
            data = request(
                EVIDENCE,
                {"efoId": ot_disease_id, "ids": group, "cursor": cursor},
                args.cache_dir / "evidence",
            )
            payload = (data.get("disease") or {}).get("evidences") or {"rows": [], "cursor": None}
            for row in payload.get("rows", []):
                ensembl = clean((row.get("target") or {}).get("id"))
                if ensembl not in inverse:
                    continue
                raw_rows.append(
                    {
                        "disease_id": disease_id,
                        "disease_name": disease_name,
                        "open_targets_disease_id": ot_disease_id,
                        "protein_id": inverse[ensembl],
                        "open_targets_target_id": ensembl,
                        "target_symbol": clean((row.get("target") or {}).get("approvedSymbol")),
                        "target_name": clean((row.get("target") or {}).get("approvedName")),
                        "datasource_id": clean(row.get("datasourceId")),
                        "evidence_score": row.get("score"),
                        "direction_on_target": clean(row.get("directionOnTarget")),
                        "direction_on_trait": clean(row.get("directionOnTrait")),
                        "evidence_id": clean(row.get("id")),
                    }
                )
            cursor = payload.get("cursor")
            if not cursor:
                break
        print(f"Retrieved Open Targets evidence batch {group_index}.", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw_rows).drop_duplicates(
        ["protein_id", "evidence_id"]
    )
    raw.to_csv(args.output_dir / "raw_target_evidence.csv", index=False)

    if raw.empty:
        summary = targets[["protein_id", "protein_label"]].copy()
        summary["open_targets_mapping_status"] = summary["protein_id"].map(
            lambda value: "MAPPED_NO_EVIDENCE" if value in protein_to_ensembl else "UNRESOLVED"
        )
        summary["evidence_record_count"] = 0
        summary["datasource_count"] = 0
        summary["max_evidence_score"] = ""
        summary["datasources"] = ""
        summary["signed_direction_record_count"] = 0
    else:
        raw["evidence_score"] = pd.to_numeric(raw["evidence_score"], errors="coerce")
        aggregate = raw.groupby("protein_id", as_index=False).agg(
            evidence_record_count=("evidence_id", "nunique"),
            datasource_count=("datasource_id", "nunique"),
            max_evidence_score=("evidence_score", "max"),
            mean_evidence_score=("evidence_score", "mean"),
            datasources=("datasource_id", lambda values: " | ".join(sorted(set(filter(None, values))))),
            signed_direction_record_count=(
                "direction_on_target",
                lambda values: int(sum(bool(clean(value)) for value in values)),
            ),
        )
        summary = targets.merge(aggregate, on="protein_id", how="left").fillna("")
        summary["open_targets_mapping_status"] = summary["protein_id"].map(
            lambda value: "MAPPED_WITH_EVIDENCE" if value in set(raw["protein_id"])
            else ("MAPPED_NO_EVIDENCE" if value in protein_to_ensembl else "UNRESOLVED")
        )

    summary["evidence_coverage_category"] = "NO_PUBLIC_ASSOCIATION_RETRIEVED"
    summary.loc[
        summary["open_targets_mapping_status"].eq("MAPPED_NO_EVIDENCE"),
        "evidence_coverage_category",
    ] = "MAPPED_NO_ASSOCIATION_RETRIEVED"
    summary.loc[
        pd.to_numeric(summary["datasource_count"], errors="coerce").fillna(0).eq(1),
        "evidence_coverage_category",
    ] = "SINGLE_DATASOURCE_CONTEXT"
    summary.loc[
        pd.to_numeric(summary["datasource_count"], errors="coerce").fillna(0).ge(2),
        "evidence_coverage_category",
    ] = "MULTI_DATASOURCE_CONTEXT"
    summary["interpretation_warning"] = (
        "Evidence coverage and source diversity indicate research context, not target causality "
        "or a direction for pharmacological intervention."
    )
    summary.sort_values(
        ["datasource_count", "evidence_record_count", "protein_label"],
        ascending=[False, False, True],
    ).to_csv(args.output_dir / "target_evidence_summary.csv", index=False)

    manifest = {
        "alzheimers_panel_proteins": int(len(targets)),
        "mapped_to_open_targets": int(len(protein_to_ensembl)),
        "open_targets_evidence_records": int(len(raw)),
        "proteins_with_retrieved_evidence": int(raw["protein_id"].nunique()) if not raw.empty else 0,
        "policy": (
            "The target panel measures source coverage and evidence diversity only. "
            "It does not derive therapeutic direction or disease benefit from these records."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote Alzheimer Open Targets panel to {args.output_dir}")


if __name__ == "__main__":
    main()
