"""Apply reviewed clinical outcomes to ReMedy mechanism-consistent candidates.

ClinicalTrials.gov retrieval creates a review queue, not an automatic efficacy claim.
This gate consumes a reviewed, source-cited adjudication table and separates
negative/harm evidence from novel mechanism hypotheses.

Required adjudication fields:
  drug_id, disease_id, clinical_outcome_verdict, source, evidence_id, source_url

Allowed verdicts:
  POSITIVE_HUMAN_EVIDENCE
  NEGATIVE_HUMAN_EVIDENCE
  DISEASE_RELEVANT_SAFETY_SIGNAL
  NO_HUMAN_EVIDENCE
  INCONCLUSIVE
  REVIEW_REQUIRED
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED = {
    "POSITIVE_HUMAN_EVIDENCE",
    "NEGATIVE_HUMAN_EVIDENCE",
    "DISEASE_RELEVANT_SAFETY_SIGNAL",
    "NO_HUMAN_EVIDENCE",
    "INCONCLUSIVE",
    "REVIEW_REQUIRED",
}


def text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    frame = pd.read_csv(path, dtype={"drug_id": str, "disease_id": str})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    for column in ("drug_id", "disease_id"):
        frame[column] = frame[column].map(text)
    return frame


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, required=True,
                        help="candidate_gate_audit.csv from evidence_gate.py")
    parser.add_argument("--adjudications", type=Path, required=True,
                        help="Reviewed clinical-outcome CSV.")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "clinical_outcome_gate")
    return parser.parse_args()


def decision(verdict: str) -> tuple[str, str]:
    verdict = text(verdict).upper()
    mapping = {
        "NEGATIVE_HUMAN_EVIDENCE": (
            "EXCLUDED_NEGATIVE_CLINICAL_EVIDENCE",
            "Human study evidence reports no benefit for the specified endpoint.",
        ),
        "DISEASE_RELEVANT_SAFETY_SIGNAL": (
            "EXCLUDED_DISEASE_RELEVANT_SAFETY_SIGNAL",
            "Source-cited disease-relevant safety signal.",
        ),
        "POSITIVE_HUMAN_EVIDENCE": (
            "CLINICAL_EVIDENCE_SUPPORTED",
            "Positive human evidence is recorded for the specified endpoint.",
        ),
        "NO_HUMAN_EVIDENCE": (
            "MECHANISTIC_HYPOTHESIS",
            "No human outcome evidence identified; retain as a novel research hypothesis.",
        ),
        "INCONCLUSIVE": (
            "INCONCLUSIVE_CLINICAL_EVIDENCE",
            "Human evidence exists but is inconclusive for the specified endpoint.",
        ),
    }
    return mapping.get(verdict, (
        "CLINICAL_OUTCOME_REVIEW_PENDING",
        "Clinical outcome has not been source-adjudicated.",
    ))


def main() -> None:
    args = parse_args()
    candidates = read_csv(args.candidate_audit, {
        "drug_id", "disease_id", "eligibility_status",
    })
    adjudications = read_csv(args.adjudications, {
        "drug_id", "disease_id", "clinical_outcome_verdict",
        "source", "evidence_id", "source_url",
    })
    adjudications["clinical_outcome_verdict"] = (
        adjudications["clinical_outcome_verdict"].map(text).str.upper()
    )
    invalid = set(adjudications["clinical_outcome_verdict"]) - ALLOWED
    if invalid:
        raise ValueError(f"Unsupported clinical-outcome verdicts: {sorted(invalid)}")
    if adjudications.duplicated(["drug_id", "disease_id"]).any():
        raise ValueError("Adjudications must contain one final verdict per drug-disease pair.")

    evidence_columns = [
        "drug_id", "disease_id", "clinical_outcome_verdict",
        "source", "evidence_id", "source_url",
    ]
    for optional in ("clinical_endpoint", "review_rationale"):
        if optional in adjudications.columns:
            evidence_columns.append(optional)

    merged = candidates.merge(
        adjudications[evidence_columns],
        on=["drug_id", "disease_id"],
        how="left",
        suffixes=("", "_clinical"),
    )
    final_status, final_reason = [], []
    for row in merged.itertuples(index=False):
        if text(row.eligibility_status) != "EVIDENCE_SUPPORTED":
            final_status.append("WITHHELD_BY_MECHANISM_GATE")
            final_reason.append("Pair did not pass the prior mechanism/evidence gate.")
        else:
            status, reason = decision(getattr(row, "clinical_outcome_verdict", ""))
            final_status.append(status)
            final_reason.append(reason)

    merged["clinical_priority_status"] = final_status
    merged["clinical_priority_reason"] = final_reason
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "clinical_outcome_gate_audit.csv", index=False)
    shortlist = merged[merged["clinical_priority_status"].isin({
        "CLINICAL_EVIDENCE_SUPPORTED", "MECHANISTIC_HYPOTHESIS",
    })].copy()
    shortlist.to_csv(args.output_dir / "researcher_hypothesis_shortlist.csv", index=False)
    summary = {
        "input_candidates": int(len(merged)),
        "researcher_shortlist": int(len(shortlist)),
        "status_counts": merged["clinical_priority_status"].value_counts().to_dict(),
        "policy": {
            "negative_human_evidence": "excluded",
            "disease_relevant_safety_signal": "excluded",
            "unreviewed_outcome": "withheld",
            "no_human_evidence": "mechanistic_hypothesis",
        },
    }
    (args.output_dir / "clinical_outcome_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
