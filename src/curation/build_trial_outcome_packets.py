"""Build endpoint-level, source-linked ClinicalTrials.gov review packets.

This is an evidence-retrieval tool, not an efficacy classifier.  It makes the
primary outcome, result availability, arm labels, and source URL available in a
compact table so a researcher can record a cited POSITIVE, NEGATIVE, or
UNRESOLVED verdict in ReMedy's evidence-chain workspace.

Run after fetch_clinical_trial_evidence.py.  The script only queries NCT IDs
already matched to bounded ReMedy candidate pairs and caches each raw record.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd


API_BASE = "https://clinicaltrials.gov/api/v2/studies"


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clinical",
        type=Path,
        default=root / "data" / "processed" / "clinical_trial_evidence.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "evidence_chain" / "trial_outcome_review_packets.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "clinicaltrials_by_nct",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def fetch_study(nct_id: str, cache_dir: Path, refresh: bool) -> dict:
    path = cache_dir / f"{nct_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        with urlopen(f"{API_BASE}/{nct_id}", timeout=30) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError, ValueError) as error:
        payload = {"_retrieval_error": str(error)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.12)
    return payload


def endpoint_rows(base: dict, study: dict) -> list[dict]:
    results = study.get("resultsSection", {})
    module = results.get("outcomeMeasuresModule", {})
    measures = module.get("outcomeMeasures", []) or []
    if not measures:
        return [{
            **base,
            "packet_status": "NO_POSTED_ENDPOINT_RESULTS",
            "outcome_type": "",
            "outcome_title": "",
            "outcome_description": "",
            "time_frame": "",
            "analysis_count": 0,
            "arm_labels": "",
            "endpoint_data_present": False,
            "review_question": (
                "No endpoint result was available in the registry record. Locate a publication "
                "or record why the study has no posted outcome."
            ),
        }]

    rows: list[dict] = []
    for measure in measures:
        groups = measure.get("groups", []) or []
        analyses = measure.get("analyses", []) or []
        group_titles = [clean(group.get("title", "")) for group in groups]
        rows.append({
            **base,
            "packet_status": "POSTED_ENDPOINT_RESULT_REVIEW_REQUIRED",
            "outcome_type": clean(measure.get("type", "")),
            "outcome_title": clean(measure.get("title", "")),
            "outcome_description": clean(measure.get("description", "")),
            "time_frame": clean(measure.get("timeFrame", "")),
            "analysis_count": len(analyses),
            "arm_labels": " | ".join(title for title in group_titles if title),
            "endpoint_data_present": True,
            "review_question": (
                "Compare the intervention arm with the relevant comparator on this endpoint; "
                "interpret direction, effect size, statistical uncertainty, and adverse events "
                "from the linked source before entering any human-outcome verdict."
            ),
        })
    return rows


def main() -> None:
    args = parse_args()
    clinical = pd.read_csv(args.clinical, dtype=str).fillna("")
    required = {"drug_id", "disease_id", "drug_name", "disease_name", "nct_id"}
    missing = required - set(clinical.columns)
    if missing:
        raise ValueError(f"{args.clinical} is missing columns: {sorted(missing)}")

    matched = clinical[clinical["nct_id"].map(clean).ne("")].copy()
    if matched.empty:
        raise ValueError("No matched NCT IDs were found in the clinical evidence table.")

    packets: list[dict] = []
    unique_studies = list(matched["nct_id"].map(clean).drop_duplicates())
    fetched: dict[str, dict] = {}
    for index, nct_id in enumerate(unique_studies, start=1):
        if index == 1 or index % 10 == 0 or index == len(unique_studies):
            print(f"Trial detail retrieval {index}/{len(unique_studies)}: {nct_id}", flush=True)
        fetched[nct_id] = fetch_study(nct_id, args.cache_dir, args.refresh)

    for row in matched.itertuples(index=False):
        nct_id = clean(row.nct_id)
        study = fetched[nct_id]
        base = {
            "drug_id": clean(row.drug_id),
            "drug_name": clean(row.drug_name),
            "disease_id": clean(row.disease_id),
            "disease_name": clean(row.disease_name),
            "nct_id": nct_id,
            "trial_title": clean(row.trial_title),
            "overall_status": clean(row.overall_status),
            "phase": clean(row.phase),
            "clinical_evidence_state": clean(row.clinical_evidence_state),
            "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
            "automated_interpretation": "NONE",
        }
        if study.get("_retrieval_error"):
            packets.append({
                **base,
                "packet_status": "TRIAL_DETAIL_RETRIEVAL_ERROR",
                "outcome_type": "",
                "outcome_title": "",
                "outcome_description": "",
                "time_frame": "",
                "analysis_count": 0,
                "arm_labels": "",
                "endpoint_data_present": False,
                "review_question": "Retry the source retrieval before making any evidence claim.",
                "retrieval_error": clean(study["_retrieval_error"]),
            })
            continue
        for packet in endpoint_rows(base, study):
            packet["retrieval_error"] = ""
            packets.append(packet)

    output = pd.DataFrame(packets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "candidate_trial_records": int(len(matched)),
        "unique_trials": int(len(unique_studies)),
        "review_packet_rows": int(len(output)),
        "packet_status_counts": output["packet_status"].value_counts().to_dict(),
        "interpretation_policy": (
            "The tool retrieves endpoint material and never classifies efficacy or harm. "
            "A source-cited researcher adjudication is required for a final evidence-chain verdict."
        ),
    }
    args.output.with_name("trial_outcome_packet_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote trial outcome review packets to {args.output}")


if __name__ == "__main__":
    main()
