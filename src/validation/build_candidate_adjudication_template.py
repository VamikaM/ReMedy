"""Create a source-linked manual adjudication sheet for ReMedy candidates.

Automated retrieval can find trials and papers, but it must not infer clinical
benefit or harm from a title, abstract, or trial registration. This script
creates a bounded review sheet for the mechanism-compatible candidate pairs.
A populated sheet can later contribute only source-cited positive or reliable
negative labels to V3 training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ALLOWED_VERDICTS = (
    "POSITIVE_HUMAN_EVIDENCE",
    "NEGATIVE_HUMAN_EVIDENCE",
    "DISEASE_RELEVANT_SAFETY_SIGNAL",
    "NO_HUMAN_EVIDENCE",
    "INCONCLUSIVE",
    "REVIEW_REQUIRED",
)


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def joined(values: pd.Series) -> str:
    return " | ".join(dict.fromkeys(clean(value) for value in values if clean(value)))


def read_csv(path: Path, required: set[str], optional: bool = False) -> pd.DataFrame:
    if not path.exists():
        if optional:
            return pd.DataFrame(columns=sorted(required))
        raise FileNotFoundError(f"Required input is missing: {path}")
    frame = pd.read_csv(path, dtype={"drug_id": str, "disease_id": str})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    for column in ("drug_id", "disease_id"):
        frame[column] = frame[column].map(clean)
    return frame


def summarise_trials(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "drug_id", "disease_id", "trial_ids", "trial_states",
            "trial_statuses", "trial_urls", "trial_outcomes",
        ])
    return frame.groupby(["drug_id", "disease_id"], as_index=False).agg(
        trial_ids=("nct_id", joined),
        trial_states=("clinical_evidence_state", joined),
        trial_statuses=("overall_status", joined),
        trial_urls=("source_url", joined),
        trial_outcomes=("outcome_titles", joined),
    )


def summarise_papers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "drug_id", "disease_id", "pubmed_ids", "pubmed_titles",
            "pubmed_urls", "publication_types",
        ])
    return frame.groupby(["drug_id", "disease_id"], as_index=False).agg(
        pubmed_ids=("evidence_id", joined),
        pubmed_titles=("article_title", joined),
        pubmed_urls=("source_url", joined),
        publication_types=("publication_types", joined),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument(
        "--clinical-trials", type=Path,
        default=root / "data" / "processed" / "clinical_trial_evidence.csv",
    )
    parser.add_argument(
        "--pubmed", type=Path,
        default=root / "data" / "processed" / "pubmed_evidence.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "reports" / "review" / "candidate_adjudication_template.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = read_csv(args.candidate_audit, {
        "drug_id", "disease_id", "drug_name", "disease_name",
        "model_score", "eligibility_status",
    })
    trials = read_csv(args.clinical_trials, {
        "drug_id", "disease_id", "nct_id", "clinical_evidence_state",
        "overall_status", "source_url", "outcome_titles",
    }, optional=True)
    papers = read_csv(args.pubmed, {
        "drug_id", "disease_id", "evidence_id", "article_title",
        "source_url", "publication_types",
    }, optional=True)

    review = audit.loc[audit["eligibility_status"].eq("EVIDENCE_SUPPORTED")].copy()
    review = review.merge(summarise_trials(trials), on=["drug_id", "disease_id"], how="left")
    review = review.merge(summarise_papers(papers), on=["drug_id", "disease_id"], how="left")

    for column in ("trial_ids", "trial_states", "trial_statuses", "trial_urls",
                   "trial_outcomes", "pubmed_ids", "pubmed_titles", "pubmed_urls",
                   "publication_types"):
        review[column] = review[column].fillna("")

    review["review_status"] = "PENDING"
    review["review_verdict"] = ""
    review["source_citation"] = ""
    review["reviewer_notes"] = ""
    review["reviewed_by"] = ""
    review["review_date"] = ""
    review["allowed_verdicts"] = " | ".join(ALLOWED_VERDICTS)

    priority_columns = [
        "drug_id", "drug_name", "disease_id", "disease_name", "model_score",
        "compatible_proteins", "compatible_drug_actions",
        "compatible_disease_effects", "safety_status", "trial_ids",
        "trial_states", "trial_statuses", "trial_urls", "trial_outcomes",
        "pubmed_ids", "pubmed_titles", "pubmed_urls", "publication_types",
        "review_status", "review_verdict", "source_citation", "reviewer_notes",
        "reviewed_by", "review_date", "allowed_verdicts",
    ]
    available = [column for column in priority_columns if column in review.columns]
    review = review[available].sort_values(["disease_name", "model_score"], ascending=[True, False])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output, index=False)
    print(f"Wrote {len(review)} source-linked review rows to {args.output}")
    print("Only source-cited adjudications should be used as V3 supervised labels.")


if __name__ == "__main__":
    main()
