"""Evaluate off-panel Phase-4 LINCS reversal drugs against Open Targets Alzheimer evidence.

Only exact ChEMBL-resolved Phase-4 perturbations are screened. The resulting
association evidence expands the review space; it never establishes efficacy,
causality, or safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
MAP = """query Map($terms:[String!]!,$entities:[String!]) { mapIds(queryTerms:$terms,entityNames:$entities) { mappings { term hits { id entity name } } } }"""
TARGETS = """query Targets($ids:[String!]!) { targets(ensemblIds:$ids) { id proteinIds { id source } approvedSymbol approvedName } }"""
EVIDENCE = """query Evidence($efoId:String!,$ids:[String!]!,$cursor:String) { disease(efoId:$efoId) { evidences(ensemblIds:$ids,size:3000,cursor:$cursor) { cursor rows { datasourceId score target { id approvedSymbol approvedName } } } } }"""


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def chunks(values: list[str], size: int = 50):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def cached_json(url: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    request = Request(url, headers={"User-Agent": "ReMedy academic evidence workflow"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def chembl_target(target_id: str, cache_dir: Path) -> dict:
    return cached_json(f"{CHEMBL}/target/{target_id}.json", cache_dir / f"{target_id}.json")


def ot_request(query: str, variables: dict, cache_dir: Path) -> dict:
    key = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    request = Request(
        OT, data=json.dumps({"query": query, "variables": variables}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "ReMedy academic evidence workflow"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload["data"]), encoding="utf-8")
            return payload["data"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution_dir = args.curated_dir / "lincs_reversal" / "compound_resolution"
    candidates = pd.read_csv(
        resolution_dir / "lincs_reversal_chembl_resolved_candidates.csv", dtype=str
    ).fillna("")
    mechanisms = pd.read_csv(
        resolution_dir / "lincs_reversal_all_chembl_mechanisms.csv", dtype=str
    ).fillna("")
    disease = pd.read_csv(args.curated_dir / "entities" / "disease.csv", dtype=str).fillna("").iloc[0]
    selected = candidates.loc[
        candidates["candidate_resolution_status"].eq(
            "LINCS_SIGNAL_PHASE4_WITHOUT_ALZHEIMER_PANEL_MECHANISM"
        )
    ].copy()
    selected_ids = set(selected["chembl_molecule_id"].map(clean)) - {""}
    if not selected_ids:
        raise RuntimeError("No resolved Phase-4 off-panel LINCS compounds available for expansion.")

    mechanisms = mechanisms.loc[mechanisms["chembl_molecule_id"].isin(selected_ids)].copy()
    target_ids = sorted(set(mechanisms["chembl_target_id"].map(clean)) - {""})
    chembl_rows = []
    for number, target_id in enumerate(target_ids, start=1):
        record = chembl_target(target_id, args.curated_dir / "raw_cache" / "off_panel_target_expansion" / "chembl")
        for component in record.get("target_components", []):
            accession = clean(component.get("accession")).upper()
            if accession:
                chembl_rows.append(
                    {
                        "chembl_target_id": target_id,
                        "uniprot_accession": accession,
                        "chembl_target_name": clean(record.get("pref_name")),
                    }
                )
        if number % 10 == 0 or number == len(target_ids):
            print(f"Retrieved {number}/{len(target_ids)} ChEMBL target records.", flush=True)
    chembl_targets = pd.DataFrame(chembl_rows).drop_duplicates()
    accessions = sorted(set(chembl_targets["uniprot_accession"])) if not chembl_targets.empty else []

    cache = args.curated_dir / "raw_cache" / "off_panel_target_expansion" / "open_targets"
    data = ot_request(
        MAP, {"terms": [clean(disease["disease_id"]), clean(disease["disease_name"])], "entities": ["disease"]}, cache
    )
    hits = [hit for row in data["mapIds"]["mappings"] for hit in row.get("hits", [])]
    disease_candidates = [h for h in hits if clean(h.get("entity")).upper() == "DISEASE"]
    expected = clean(disease["disease_id"]).replace(":", "_")
    ot_disease = next((clean(h["id"]) for h in disease_candidates if clean(h.get("id")) == expected), "")
    if not ot_disease:
        raise RuntimeError("Could not map Alzheimer disease to Open Targets.")

    hit_map: dict[str, list[dict]] = {}
    for group in chunks(accessions):
        data = ot_request(MAP, {"terms": group, "entities": ["target"]}, cache)
        for row in data["mapIds"]["mappings"]:
            hit_map[clean(row["term"]).upper()] = row.get("hits", [])
    ensembl_ids = sorted({
        clean(hit.get("id")) for hits in hit_map.values() for hit in hits
        if clean(hit.get("entity")).upper() == "TARGET" and clean(hit.get("id"))
    })
    protein_ids: dict[str, set[str]] = {}
    target_symbols: dict[str, str] = {}
    for group in chunks(ensembl_ids):
        data = ot_request(TARGETS, {"ids": group}, cache)
        for target in data.get("targets", []):
            if target:
                target_id = clean(target["id"])
                protein_ids[target_id] = {
                    clean(p.get("id")).upper() for p in target.get("proteinIds", []) if clean(p.get("id"))
                }
                target_symbols[target_id] = clean(target.get("approvedSymbol"))

    accession_to_ensembl: dict[str, str] = {}
    for accession in accessions:
        exact = [
            clean(hit.get("id")) for hit in hit_map.get(accession, [])
            if clean(hit.get("entity")).upper() == "TARGET"
            and accession in protein_ids.get(clean(hit.get("id")), set())
        ]
        if len(set(exact)) == 1:
            accession_to_ensembl[accession] = exact[0]

    evidence_rows = []
    for group in chunks(sorted(set(accession_to_ensembl.values()))):
        cursor = None
        while True:
            data = ot_request(EVIDENCE, {"efoId": ot_disease, "ids": group, "cursor": cursor}, cache)
            page = data.get("disease", {}).get("evidences", {})
            for row in page.get("rows", []):
                evidence_rows.append(
                    {
                        "ensembl_id": clean(row.get("target", {}).get("id")),
                        "target_symbol": clean(row.get("target", {}).get("approvedSymbol")),
                        "datasource": clean(row.get("datasourceId")),
                        "score": row.get("score"),
                    }
                )
            cursor = page.get("cursor")
            if not cursor:
                break

    evidence = pd.DataFrame(evidence_rows)
    if evidence.empty:
        summary = pd.DataFrame(columns=["ensembl_id", "target_symbol", "ot_evidence_count", "ot_datasource_count", "ot_datasources"])
    else:
        summary = evidence.groupby(["ensembl_id", "target_symbol"], as_index=False).agg(
            ot_evidence_count=("datasource", "count"),
            ot_datasource_count=("datasource", lambda x: len({clean(v) for v in x if clean(v)})),
            ot_datasources=("datasource", lambda x: " | ".join(sorted({clean(v) for v in x if clean(v)}))),
        )
    chembl_targets["ensembl_id"] = chembl_targets["uniprot_accession"].map(accession_to_ensembl)
    target_context = chembl_targets.merge(summary, on="ensembl_id", how="left").fillna("")
    expanded = (
        mechanisms.merge(target_context, on="chembl_target_id", how="left")
        .groupby("chembl_molecule_id", as_index=False)
        .agg(
            expanded_targets=("target_symbol", lambda x: " | ".join(sorted({clean(v) for v in x if clean(v)}))),
            expanded_target_evidence_count=("ot_evidence_count", lambda x: sum(pd.to_numeric(x, errors="coerce").fillna(0))),
            expanded_target_datasources=("ot_datasources", lambda x: " | ".join(sorted({clean(v) for v in x if clean(v)}))),
        )
    )
    result = selected.merge(expanded, on="chembl_molecule_id", how="left").fillna("")
    result["expansion_status"] = result["expanded_target_evidence_count"].map(
        lambda x: "OFF_PANEL_TARGETS_WITH_OPEN_TARGETS_AD_EVIDENCE"
        if float(x or 0) > 0 else "OFF_PANEL_TARGETS_WITHOUT_RETRIEVED_AD_EVIDENCE"
    )
    output = args.curated_dir / "lincs_reversal" / "off_panel_expansion"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "phase4_lincs_off_panel_target_expansion.csv", index=False)
    target_context.to_csv(output / "off_panel_target_open_targets_evidence.csv", index=False)
    print(json.dumps({
        "phase4_off_panel_lincs_compounds": len(result),
        "unique_off_panel_chembl_targets": len(target_ids),
        "uniprot_targets_mapped_to_open_targets": len(accession_to_ensembl),
        "compounds_with_retrieved_ad_target_evidence": int(result["expansion_status"].eq("OFF_PANEL_TARGETS_WITH_OPEN_TARGETS_AD_EVIDENCE").sum()),
        "output": str(output / "phase4_lincs_off_panel_target_expansion.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
