"""Collect signed Open Targets evidence only for mechanism-supported disease proteins."""

from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
MAP = """query Map($terms: [String!]!, $entities: [String!]) { mapIds(queryTerms: $terms, entityNames: $entities) { mappings { term hits { id entity name } } } }"""
TARGETS = """query Targets($ids: [String!]!) { targets(ensemblIds: $ids) { id proteinIds { id source } } }"""
EVIDENCE = """query Evidence($efoId: String!, $ids: [String!]!, $cursor: String) { disease(efoId: $efoId) { evidences(ensemblIds: $ids, size: 3000, cursor: $cursor) { cursor rows { id datasourceId score directionOnTarget directionOnTrait target { id } } } } }"""

def clean(value):
    return "" if pd.isna(value) else str(value).strip()

def normalise(value):
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()

def chunks(values, size=250):
    for start in range(0, len(values), size):
        yield values[start:start + size]

def request(query, variables, cache_dir):
    fingerprint = hashlib.sha256(json.dumps([query, variables], sort_keys=True).encode()).hexdigest()
    cached = cache_dir / f"{fingerprint}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = Request(ENDPOINT, data=body, headers={"Content-Type": "application/json", "User-Agent": "ReMedy academic research pipeline"}, method="POST")
    for attempt in range(4):
        try:
            with urlopen(req, timeout=60) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(payload["data"]), encoding="utf-8")
            return payload["data"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def effect(on_target, on_trait):
    target = clean(on_target).upper()
    trait = clean(on_trait).upper()
    a = 1 if target in {"GOF", "GAIN_OF_FUNCTION"} else -1 if target in {"LOF", "LOSS_OF_FUNCTION"} else 0
    b = 1 if trait == "RISK" else -1 if trait == "PROTECTIVE" else 0
    return "INCREASED_ACTIVITY" if a * b == 1 else "DECREASED_ACTIVITY" if a * b == -1 else ""

def args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=root / "data/processed/nodes.parquet")
    parser.add_argument("--edges", type=Path, default=root / "data/processed/edges.parquet")
    parser.add_argument("--mechanisms", type=Path, default=root / "data/processed/drug_mechanisms.csv")
    parser.add_argument("--output", type=Path, default=root / "data/processed/disease_target_effects.csv")
    parser.add_argument("--audit-dir", type=Path, default=root / "reports/evidence_gate")
    parser.add_argument("--cache-dir", type=Path, default=root / "data/raw/open_targets_evidence")
    return parser.parse_args()

def map_disease(node_id, label, hits):
    candidates = {clean(hit.get("id")): hit for hit in hits if clean(hit.get("entity")).upper() == "DISEASE" and clean(hit.get("id"))}
    values = list(candidates.values())
    if node_id.upper().startswith("EFO_"):
        return next((hit for hit in values if clean(hit.get("id")) == node_id), None)
    exact = [hit for hit in values if normalise(hit.get("name")) == normalise(label)]
    return exact[0] if len(exact) == 1 else None

def main():
    cfg = args()
    nodes, edges = pd.read_parquet(cfg.nodes), pd.read_parquet(cfg.edges)
    mechanisms = pd.read_csv(cfg.mechanisms, dtype={"protein_id": str})
    required = {"drug_id", "protein_id", "action_type", "source", "evidence_id"}
    if required - set(mechanisms.columns):
        raise ValueError(f"drug_mechanisms.csv missing {sorted(required - set(mechanisms.columns))}")
    if mechanisms.empty:
        raise RuntimeError("No graph-mapped ChEMBL mechanisms were collected. Fix mechanism ID mapping before querying Open Targets.")

    diseases = nodes.loc[nodes["node_type"].astype(str).str.lower().eq("disease"), ["node_id", "label"]].copy()
    diseases["node_id"] = diseases["node_id"].map(clean)
    disease_ids = set(diseases["node_id"])
    edges["source_id"], edges["target_id"] = edges["source_id"].map(clean), edges["target_id"].map(clean)
    edges["rel"] = edges["relation"].map(lambda value: clean(value).lower())
    mechanism_proteins = set(mechanisms["protein_id"].map(clean))
    assoc = edges[(edges["rel"] == "associated_with") & edges["source_id"].isin(disease_ids) & edges["target_id"].isin(mechanism_proteins)]
    proteins_by_disease = assoc.groupby("source_id")["target_id"].agg(lambda x: sorted(set(x))).to_dict()
    needed = sorted({protein for values in proteins_by_disease.values() for protein in values})
    if not needed:
        raise RuntimeError("No protein is both disease-associated and ChEMBL-mechanism-supported; no Open Targets query started.")
    def acc(node_id): return node_id.split(":")[-1].upper()
    accession_to_protein = {acc(protein): protein for protein in needed}
    print(f"Open Targets scope: {len(diseases)} diseases, {len(needed)} mechanism-and-disease proteins.", flush=True)

    disease_terms = list(dict.fromkeys(diseases["node_id"].tolist() + diseases["label"].map(clean).tolist()))
    disease_hits = {}
    for group in chunks(disease_terms):
        data = request(MAP, {"terms": group, "entities": ["disease"]}, cfg.cache_dir)
        disease_hits.update({clean(x["term"]): x.get("hits", []) for x in data["mapIds"]["mappings"]})
    disease_to_efo, audit = {}, []
    for row in diseases.itertuples(index=False):
        selected = map_disease(row.node_id, row.label, disease_hits.get(row.node_id, []) + disease_hits.get(clean(row.label), []))
        disease_to_efo[row.node_id] = clean(selected.get("id")) if selected else ""
        audit.append({"entity_type":"disease","remedy_id":row.node_id,"label":row.label,"open_targets_id":disease_to_efo[row.node_id],"mapping_status":"MAPPED_EXACT" if selected else "UNRESOLVED"})

    target_hits = {}
    for group in chunks(sorted(accession_to_protein)):
        data = request(MAP, {"terms": group, "entities": ["target"]}, cfg.cache_dir)
        target_hits.update({clean(x["term"]): x.get("hits", []) for x in data["mapIds"]["mappings"]})
    possible_ids = sorted({clean(hit.get("id")) for hits in target_hits.values() for hit in hits if clean(hit.get("entity")).upper() == "TARGET" and clean(hit.get("id"))})
    target_accessions = {}
    for group in chunks(possible_ids):
        data = request(TARGETS, {"ids": group}, cfg.cache_dir)
        for target in data.get("targets", []):
            if target:
                target_accessions[clean(target["id"])] = {clean(item.get("id")).upper() for item in target.get("proteinIds", [])}

    protein_to_ensembl = {}
    for accession, protein in accession_to_protein.items():
        candidates = sorted({clean(hit.get("id")) for hit in target_hits.get(accession, []) if clean(hit.get("entity")).upper() == "TARGET" and accession in target_accessions.get(clean(hit.get("id")), set())})
        if len(candidates) == 1:
            protein_to_ensembl[protein] = candidates[0]
            status, target_id = "MAPPED_VERIFIED", candidates[0]
        else:
            status, target_id = ("UNRESOLVED" if not candidates else "AMBIGUOUS"), ""
        audit.append({"entity_type":"protein","remedy_id":protein,"label":accession,"open_targets_id":target_id,"mapping_status":status})

    evidence = []
    for disease_id, efo_id in disease_to_efo.items():
        mapping = {protein_to_ensembl[p]: p for p in proteins_by_disease.get(disease_id, []) if p in protein_to_ensembl}
        if not efo_id or not mapping:
            continue
        print(f"Querying {disease_id}: {len(mapping)} mapped disease proteins.", flush=True)
        for group in chunks(sorted(mapping)):
            cursor = None
            while True:
                data = request(EVIDENCE, {"efoId":efo_id, "ids":group, "cursor":cursor}, cfg.cache_dir)
                payload = (data.get("disease") or {}).get("evidences") or {"rows": [], "cursor": None}
                for row in payload["rows"]:
                    signed_effect = effect(row.get("directionOnTarget"), row.get("directionOnTrait"))
                    target_id = clean((row.get("target") or {}).get("id"))
                    if signed_effect and target_id in mapping:
                        evidence.append({"disease_id":disease_id,"protein_id":mapping[target_id],"disease_effect":signed_effect,"source":f"OpenTargets:{clean(row.get('datasourceId'))}","evidence_id":f"OpenTargets:{clean(row.get('id'))}","direction_on_target":clean(row.get("directionOnTarget")),"direction_on_trait":clean(row.get("directionOnTrait")),"evidence_score":row.get("score"),"open_targets_disease_id":efo_id,"open_targets_target_id":target_id})
                cursor = payload.get("cursor")
                if not cursor:
                    break
    columns=["disease_id","protein_id","disease_effect","source","evidence_id","direction_on_target","direction_on_trait","evidence_score","open_targets_disease_id","open_targets_target_id"]
    out = pd.DataFrame(evidence, columns=columns).drop_duplicates(["disease_id","protein_id","evidence_id"])
    cfg.output.parent.mkdir(parents=True, exist_ok=True); cfg.audit_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.output, index=False); pd.DataFrame(audit).to_csv(cfg.audit_dir / "open_targets_id_mapping_audit.csv", index=False)
    print(f"Wrote {len(out):,} signed evidence records. Verified {len(protein_to_ensembl)}/{len(needed)} proteins; mapped {sum(bool(x) for x in disease_to_efo.values())}/{len(disease_to_efo)} diseases.", flush=True)

if __name__ == "__main__":
    main()
