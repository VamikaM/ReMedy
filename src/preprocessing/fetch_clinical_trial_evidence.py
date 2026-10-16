"""Retrieve auditable ClinicalTrials.gov evidence for shortlisted ReMedy pairs.

This collector deliberately does *not* infer that a registered trial demonstrates
benefit or harm.  Trial outcomes require endpoint-aware review.  It retrieves
candidate-matched study records, preserves the ClinicalTrials.gov identifiers and
outcome titles, and creates a small review queue for a researcher or an
evidence-adjudication module.

Run this only after candidate generation, on a short candidate list rather than
the full drug-disease Cartesian product.
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


API_URL = "https://clinicaltrials.gov/api/v2/studies"


def clean(value: object) -> str:
    value = re.sub(r"\\b([A-Za-z0-9]+)\\'s\\b", r"\\1", str(value))
    return re.sub(r"[^a-z0-9 ]+", " ", value.lower()).strip()


def read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"drug_id": str, "disease_id": str})
    required = {"drug_id", "disease_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.drop_duplicates(["drug_id", "disease_id"]).copy()


def fetch_pair(drug_name: str, disease_name: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params = {
        "query.intr": drug_name,
        "query.cond": disease_name,
        "pageSize": 100,
        "format": "json",
    }
    try:
        with urlopen(f"{API_URL}?{urlencode(params)}", timeout=30) as response:
            payload = json.load(response)
    except Exception as error:
        payload = {"_retrieval_error": str(error), "studies": []}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.12)
    return payload


def protocol_study_fields(study: dict) -> dict:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {}).get("conditions", []) or []
    interventions = (
        protocol.get("armsInterventionsModule", {}).get("interventions", []) or []
    )
    return {
        "nct_id": identification.get("nctId", ""),
        "title": identification.get("briefTitle", ""),
        "official_title": identification.get("officialTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "phases": " | ".join(design.get("phases", []) or []),
        "conditions": " | ".join(conditions),
        "interventions": " | ".join(
            item.get("name", "") for item in interventions if item.get("name")
        ),
    }


def is_pair_match(fields: dict, drug_name: str, disease_name: str) -> bool:
    drug = clean(drug_name)
    disease = clean(disease_name)
    intervention_text = clean(fields["interventions"] + " " + fields["title"])
    condition_text = clean(
        fields["conditions"] + " " + fields["title"] + " " + fields["official_title"]
    )
    return drug in intervention_text and disease in condition_text


def outcome_titles(study: dict) -> str:
    outcomes = (
        study.get("resultsSection", {})
        .get("outcomeMeasuresModule", {})
        .get("outcomeMeasures", [])
        or []
    )
    titles = [str(item.get("title", "")).strip() for item in outcomes]
    return " | ".join(title for title in titles if title)


def record_for_trial(
    pair: object, fields: dict, study: dict, drug_name: str, disease_name: str
) -> dict:
    has_results = bool(study.get("hasResults") or study.get("resultsSection"))
    if has_results:
        state = "POSTED_RESULTS_REVIEW_REQUIRED"
    elif fields["overall_status"] == "COMPLETED":
        state = "COMPLETED_NO_POSTED_RESULTS"
    elif fields["overall_status"] in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}:
        state = "NONCOMPLETED_TRIAL_REVIEW_REQUIRED"
    else:
        state = "TRIAL_IN_PROGRESS_OR_UNKNOWN"
    return {
        "drug_id": pair.drug_id,
        "drug_name": drug_name,
        "disease_id": pair.disease_id,
        "disease_name": disease_name,
        "clinical_evidence_state": state,
        "clinical_outcome_verdict": "REVIEW_REQUIRED",
        "nct_id": fields["nct_id"],
        "trial_title": fields["title"],
        "overall_status": fields["overall_status"],
        "phase": fields["phases"],
        "conditions": fields["conditions"],
        "interventions": fields["interventions"],
        "outcome_titles": outcome_titles(study),
        "has_posted_results": has_results,
        "source": "ClinicalTrials.gov",
        "evidence_id": f"ClinicalTrials.gov:{fields['nct_id']}",
        "source_url": f"https://clinicaltrials.gov/study/{fields['nct_id']}",
        "retrieval_query": f"intervention={drug_name}; condition={disease_name}",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="CSV containing drug_id,disease_id; model_score is optional.")
    parser.add_argument("--nodes", type=Path,
                        default=root / "data/processed/nodes.parquet")
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/clinical_trial_evidence.csv")
    parser.add_argument("--review-queue", type=Path,
                        default=root / "reports/evidence_gate/clinical_trial_review_queue.csv")
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "data/raw/clinicaltrials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pairs)
    nodes = pd.read_parquet(args.nodes)
    labels = nodes.set_index("node_id")["label"].astype(str).to_dict()

    records: list[dict] = []
    total = len(pairs)
    for index, pair in enumerate(pairs.itertuples(index=False), start=1):
        drug_name = labels.get(pair.drug_id, "")
        disease_name = labels.get(pair.disease_id, "")
        if index == 1 or index % 10 == 0 or index == total:
            print(f"Clinical trial retrieval {index}/{total}: {drug_name} -> {disease_name}", flush=True)
        if not drug_name or not disease_name:
            records.append({
                "drug_id": pair.drug_id, "drug_name": drug_name,
                "disease_id": pair.disease_id, "disease_name": disease_name,
                "clinical_evidence_state": "ENTITY_NAME_UNAVAILABLE",
                "clinical_outcome_verdict": "REVIEW_REQUIRED",
                "nct_id": "", "trial_title": "", "overall_status": "", "phase": "",
                "conditions": "", "interventions": "", "outcome_titles": "",
                "has_posted_results": False, "source": "ClinicalTrials.gov",
                "evidence_id": "", "source_url": "", "retrieval_query": "",
            })
            continue

        key = hashlib.sha256(
            f"{pair.drug_id}|{pair.disease_id}".encode("utf-8")
        ).hexdigest()[:20]
        payload = fetch_pair(drug_name, disease_name, args.cache_dir / f"{key}.json")
        studies = payload.get("studies", [])
        matched = []
        for study in studies:
            fields = protocol_study_fields(study)
            if fields["nct_id"] and is_pair_match(fields, drug_name, disease_name):
                matched.append(record_for_trial(pair, fields, study, drug_name, disease_name))

        if matched:
            records.extend(matched)
        else:
            records.append({
                "drug_id": pair.drug_id, "drug_name": drug_name,
                "disease_id": pair.disease_id, "disease_name": disease_name,
                "clinical_evidence_state": (
                    "RETRIEVAL_ERROR" if payload.get("_retrieval_error")
                    else "NO_MATCHING_REGISTERED_TRIAL"
                ),
                "clinical_outcome_verdict": "REVIEW_REQUIRED",
                "nct_id": "", "trial_title": "", "overall_status": "", "phase": "",
                "conditions": "", "interventions": "", "outcome_titles": "",
                "has_posted_results": False, "source": "ClinicalTrials.gov",
                "evidence_id": "", "source_url": "",
                "retrieval_query": f"intervention={drug_name}; condition={disease_name}",
            })

    evidence = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.review_queue.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output, index=False)
    review_queue = evidence[evidence["clinical_evidence_state"].isin({
        "POSTED_RESULTS_REVIEW_REQUIRED",
        "COMPLETED_NO_POSTED_RESULTS",
        "NONCOMPLETED_TRIAL_REVIEW_REQUIRED",
    })].copy()
    review_queue.to_csv(args.review_queue, index=False)
    print(evidence["clinical_evidence_state"].value_counts().to_string())
    print(f"Wrote {len(evidence):,} ClinicalTrials.gov evidence records to {args.output}")
    print(f"Wrote {len(review_queue):,} trial records requiring outcome review to {args.review_queue}")


if __name__ == "__main__":
    main()
