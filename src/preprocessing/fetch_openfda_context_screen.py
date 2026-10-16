"""Create a reproducible openFDA disease-context label screen for ReMedy.

For each scored drug-disease pair, this script retrieves the public US label once
per drug, caches the raw response, and checks disease names/aliases against label
contraindications and warnings.  The output is an *evidence-coverage screen*:
NO_MATCHING_CONTEXT_FLAG means the retrieved label contains no direct text match;
it is deliberately not interpreted as a clinical safety clearance.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://api.fda.gov/drug/label.json"


def normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(value).lower()).strip()


def disease_terms(label: str, aliases: list[str]) -> list[str]:
    terms = {normalise(label)}
    for alias in aliases:
        if alias:
            terms.add(normalise(alias))
    removable = {"disease", "disorder", "syndrome"}
    words = normalise(label).split()
    shortened = " ".join(word for word in words if word not in removable)
    if len(shortened) >= 4:
        terms.add(shortened)
    return sorted(term for term in terms if len(term) >= 4)


def fetch_label(drug_name: str, cache_path: Path, timeout_seconds: float) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    # Drug names are identifiers for retrieval only. The response remains the
    # authoritative source captured in the cache for later auditing.
    query = quote(f'openfda.generic_name:"{drug_name}"')
    url = f"{BASE_URL}?search={query}&limit=100"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except Exception as error:
        payload = {"_retrieval_error": str(error), "results": []}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.15)
    return payload


def label_text(result: dict, fields: list[str]) -> str:
    values: list[str] = []
    for field in fields:
        values.extend(result.get(field, []) or [])
    return normalise(" ".join(values))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path,
                        default=root / "reports" / "model_evaluation" / "hybrid_rgcn_raw_internal_scores.csv")
    parser.add_argument("--nodes", type=Path, default=root / "data" / "processed" / "nodes.parquet")
    parser.add_argument("--aliases", type=Path, default=None,
                        help="Optional CSV: disease_id,alias")
    parser.add_argument("--output", type=Path,
                        default=root / "data" / "processed" / "drug_safety_context.csv")
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "data" / "raw" / "openfda_labels")
    parser.add_argument("--timeout-seconds", type=float, default=20.0,
                        help="Maximum wait per openFDA request; failed requests are recorded for audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking = pd.read_csv(args.ranking, dtype={"drug_id": str, "disease_id": str})
    nodes = pd.read_parquet(args.nodes)
    node_labels = nodes.set_index("node_id")["label"].astype(str).to_dict()
    aliases: dict[str, list[str]] = {}
    if args.aliases and args.aliases.exists():
        alias_frame = pd.read_csv(args.aliases, dtype={"disease_id": str, "alias": str})
        aliases = alias_frame.groupby("disease_id")["alias"].agg(list).to_dict()

    cache: dict[str, dict] = {}
    unique_drug_count = ranking["drug_id"].nunique()
    processed_drug_count = 0
    output = []
    warning_fields = ["contraindications", "boxed_warning", "warnings", "warnings_and_cautions"]
    for row in ranking[["drug_id", "disease_id"]].itertuples(index=False):
        drug_name = node_labels.get(row.drug_id, "")
        disease_name = node_labels.get(row.disease_id, "")
        if not drug_name or not disease_name:
            output.append({
                "drug_id": row.drug_id, "disease_id": row.disease_id,
                "safety_status": "NOT_AVAILABLE", "source": "openFDA",
                "evidence_id": "", "matched_term": "", "matched_section": "",
            })
            continue
        if row.drug_id not in cache:
            processed_drug_count += 1
            if processed_drug_count == 1 or processed_drug_count % 25 == 0 or processed_drug_count == unique_drug_count:
                print(f"Processing drug label {processed_drug_count}/{unique_drug_count}: {drug_name}", flush=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", drug_name)
            cache[row.drug_id] = fetch_label(
                drug_name, args.cache_dir / f"{safe_name}.json", args.timeout_seconds
            )
        results = cache[row.drug_id].get("results", [])
        if not results:
            output.append({
                "drug_id": row.drug_id, "disease_id": row.disease_id,
                "safety_status": "NOT_AVAILABLE", "source": "openFDA",
                "evidence_id": "", "matched_term": "", "matched_section": "",
            })
            continue

        terms = disease_terms(disease_name, aliases.get(row.disease_id, []))
        match = None
        for result_index, result in enumerate(results):
            combined = label_text(result, warning_fields)
            term = next((item for item in terms if item in combined), None)
            if term:
                section = next(
                    (field for field in warning_fields if term in label_text(result, [field])),
                    "warning_text",
                )
                match = (term, section, str(result_index))
                break
        if match:
            status, matched_term, section, evidence_id = "FLAGGED", match[0], match[1], f"openFDA-label:{match[2]}"
        else:
            status, matched_term, section, evidence_id = "NO_MATCHING_CONTEXT_FLAG", "", "", "openFDA-label"
        output.append({
            "drug_id": row.drug_id, "disease_id": row.disease_id,
            "safety_status": status, "source": "openFDA",
            "evidence_id": evidence_id, "matched_term": matched_term,
            "matched_section": section,
        })

    frame = pd.DataFrame(output).drop_duplicates(["drug_id", "disease_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame["safety_status"].value_counts().to_string())
    print(f"Wrote {len(frame):,} disease-context label-screen records to {args.output}")


if __name__ == "__main__":
    main()
