"""Attach retrieved ClinicalTrials.gov and PubMed records to ReMedy review tables.

This script only pre-fills citations and review context. It never infers efficacy
from a trial registration, title, abstract, or keyword match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def join_unique(values: pd.Series, limit: int = 5) -> str:
    output = []
    for value in values:
        value = clean(value)
        if value and value not in output:
            output.append(value)
        if len(output) >= limit:
            break
    return " | ".join(output)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1",
    )
    parser.add_argument(
        "--clinical", type=Path,
        default=root / "data" / "processed" / "clinical_trial_evidence.csv",
    )
    parser.add_argument(
        "--pubmed", type=Path,
        default=root / "data" / "processed" / "pubmed_evidence.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes_path = args.workspace / "phenotype_human_outcome_review.csv"
    outcomes = pd.read_csv(outcomes_path, dtype=str).fillna("")
    clinical = pd.read_csv(args.clinical, dtype={"drug_id": str, "disease_id": str}).fillna("")
    pubmed = pd.read_csv(args.pubmed, dtype={"drug_id": str, "disease_id": str}).fillna("")

    for frame in (outcomes, clinical, pubmed):
        for column in ("drug_id", "disease_id"):
            frame[column] = frame[column].map(clean)

    clinical_groups = {
        pair: group
        for pair, group in clinical.groupby(["drug_id", "disease_id"], sort=False)
    }
    pubmed_groups = {
        pair: group
        for pair, group in pubmed.groupby(["drug_id", "disease_id"], sort=False)
    }

    notes, citations, verdicts = [], [], []
    for row in outcomes.itertuples(index=False):
        pair = (clean(row.drug_id), clean(row.disease_id))
        trial_rows = clinical_groups.get(pair)
        paper_rows = pubmed_groups.get(pair)
        parts, links = [], []

        existing_verdict = clean(row.human_outcome_verdict) or "PENDING"
        verdict = existing_verdict
        if trial_rows is not None and not trial_rows.empty:
            states = join_unique(trial_rows["clinical_evidence_state"])
            trials = join_unique(trial_rows["nct_id"])
            statuses = join_unique(trial_rows["overall_status"])
            titles = join_unique(trial_rows["trial_title"], limit=2)
            parts.append(
                f"ClinicalTrials.gov retrieved: state={states}; NCT={trials}; status={statuses}; title={titles}"
            )
            links.append(join_unique(trial_rows["source_url"]))
            if existing_verdict == "PENDING" and any(
                state in {
                    "POSTED_RESULTS_REVIEW_REQUIRED",
                    "COMPLETED_NO_POSTED_RESULTS",
                    "NONCOMPLETED_TRIAL_REVIEW_REQUIRED",
                }
                for state in trial_rows["clinical_evidence_state"].map(clean)
            ):
                verdict = "TRIAL_OUTCOME_UNRESOLVED"

        if paper_rows is not None and not paper_rows.empty:
            pmids = join_unique(paper_rows["evidence_id"])
            titles = join_unique(paper_rows["article_title"], limit=3)
            parts.append(f"PubMed retrieved: {pmids}; examples={titles}")
            links.append(join_unique(paper_rows["source_url"]))

        prior_note = clean(row.review_notes)
        notes.append(" || ".join(part for part in [prior_note, *parts] if part))
        prior_citation = clean(row.source_citation)
        citations.append(" | ".join(
            part for part in [prior_citation, *links] if part
        ))
        verdicts.append(verdict)

    outcomes["human_outcome_verdict"] = verdicts
    outcomes["source_citation"] = citations
    outcomes["review_notes"] = notes
    outcomes.to_csv(outcomes_path, index=False)
    print(
        outcomes["human_outcome_verdict"].value_counts(dropna=False).to_string()
    )
    print(f"Prefilled retrieval context in {outcomes_path}")


if __name__ == "__main__":
    main()
