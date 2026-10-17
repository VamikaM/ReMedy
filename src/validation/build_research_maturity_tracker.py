"""Track the source-backed research maturity of ReMedy drug-disease hypotheses.

Graph plausibility and research progress are deliberately separated. Literature
and trial records are progress signals, not automatic evidence of benefit.
Only a separate source-cited outcome adjudication may label human benefit/harm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

PASS = "EVIDENCE_SUPPORTED"
REVIEW_TRIAL = {
    "POSTED_RESULTS_REVIEW_REQUIRED",
    "COMPLETED_NO_POSTED_RESULTS",
    "NONCOMPLETED_TRIAL_REVIEW_REQUIRED",
}
ACTIVE_TRIAL = {"TRIAL_IN_PROGRESS_OR_UNKNOWN"}
ALLOWED = {
    "POSITIVE_HUMAN_EVIDENCE", "NEGATIVE_HUMAN_EVIDENCE",
    "DISEASE_RELEVANT_SAFETY_SIGNAL", "NO_HUMAN_EVIDENCE",
    "INCONCLUSIVE", "REVIEW_REQUIRED",
}

def clean(value):
    return "" if pd.isna(value) else str(value).strip()

def upper(value):
    return clean(value).upper()

def joined(values):
    return " | ".join(dict.fromkeys(clean(v) for v in values if clean(v)))

def read(path, required, optional=False):
    if path is None or not path.exists():
        if optional:
            return pd.DataFrame(columns=sorted(required))
        raise FileNotFoundError(f"Required input is missing: {path}")
    frame = pd.read_csv(path, dtype={"drug_id": str, "disease_id": str})
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    for column in ("drug_id", "disease_id"):
        frame[column] = frame[column].map(clean)
    return frame

def summarise_trials(frame):
    columns = ["drug_id", "disease_id", "clinical_states", "registered_trial_count",
               "trial_ids", "trial_statuses", "needs_outcome_review", "trial_active"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    frame["clinical_evidence_state"] = frame["clinical_evidence_state"].map(upper)
    frame["nct_id"] = frame["nct_id"].map(clean)
    group = frame.groupby(["drug_id", "disease_id"])
    result = group.agg(
        clinical_states=("clinical_evidence_state", joined),
        trial_ids=("nct_id", joined),
        trial_statuses=("overall_status", joined),
    ).reset_index()
    result["registered_trial_count"] = group["nct_id"].apply(
        lambda values: len({value for value in values if value})
    ).to_numpy()
    states = result["clinical_states"].map(lambda value: set(clean(value).split(" | ")))
    result["needs_outcome_review"] = states.map(lambda value: bool(value & REVIEW_TRIAL))
    result["trial_active"] = states.map(lambda value: bool(value & ACTIVE_TRIAL))
    return result[columns]

def summarise_papers(frame):
    columns = ["drug_id", "disease_id", "pubmed_record_count", "pubmed_ids",
               "publication_types", "literature_retrieved"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    frame["pmid"] = frame["pmid"].map(clean)
    group = frame.groupby(["drug_id", "disease_id"])
    result = group.agg(
        pubmed_ids=("pmid", joined),
        publication_types=("publication_types", joined),
    ).reset_index()
    result["pubmed_record_count"] = group["pmid"].apply(
        lambda values: len({value for value in values if value})
    ).to_numpy()
    result["literature_retrieved"] = result["pubmed_record_count"].gt(0)
    return result[columns]

def stage(row):
    if clean(row.eligibility_status) != PASS:
        return 0, "M0_WITHHELD", "WITHHELD", (
            "Do not prioritise: the pair did not pass the mechanism and context gate."
        )
    verdict = upper(row.clinical_outcome_verdict)
    if verdict == "NEGATIVE_HUMAN_EVIDENCE":
        return 4, "M4_OUTCOME_ADJUDICATED", "EXCLUDED_NEGATIVE_HUMAN_EVIDENCE", (
            "Do not promote: retain the source-cited negative human outcome for audit."
        )
    if verdict == "DISEASE_RELEVANT_SAFETY_SIGNAL":
        return 4, "M4_OUTCOME_ADJUDICATED", "EXCLUDED_DISEASE_RELEVANT_SAFETY_SIGNAL", (
            "Do not promote: retain the disease-relevant safety signal for audit."
        )
    if verdict == "POSITIVE_HUMAN_EVIDENCE":
        return 4, "M4_OUTCOME_ADJUDICATED", "POSITIVE_HUMAN_EVIDENCE", (
            "Compare endpoint-specific human evidence with current treatment standards."
        )
    if verdict == "INCONCLUSIVE":
        return 4, "M4_OUTCOME_ADJUDICATED", "INCONCLUSIVE_HUMAN_EVIDENCE", (
            "Assess endpoint relevance and whether confirmatory work is justified."
        )
    if verdict == "NO_HUMAN_EVIDENCE":
        return 1, "M1_MECHANISTIC_HYPOTHESIS", "NOVEL_MECHANISTIC_HYPOTHESIS", (
            "Plan disease-relevant preclinical or translational validation."
        )
    if bool(row.needs_outcome_review):
        return 3, "M3_HUMAN_RESEARCH_IDENTIFIED", "OUTCOME_REVIEW_REQUIRED", (
            "Review trial endpoint, population, comparator, completion reason, and outcome."
        )
    if bool(row.trial_active):
        return 3, "M3_HUMAN_RESEARCH_IDENTIFIED", "RESEARCH_IN_PROGRESS_OR_UNRESOLVED", (
            "Monitor the registered study; do not infer benefit before outcomes are available."
        )
    if bool(row.literature_retrieved):
        return 2, "M2_LITERATURE_CONTEXT_RETRIEVED", "LITERATURE_RELEVANCE_REVIEW_REQUIRED", (
            "Classify papers as direct human, preclinical, review, or indirect context."
        )
    return 1, "M1_MECHANISTIC_HYPOTHESIS", "NOVEL_MECHANISTIC_HYPOTHESIS", (
        "Retrieve disease-specific literature and design target/pathway validation."
    )

def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--clinical-trials", type=Path,
                        default=root / "data/processed/clinical_trial_evidence.csv")
    parser.add_argument("--pubmed", type=Path,
                        default=root / "data/processed/pubmed_evidence.csv")
    parser.add_argument("--adjudications", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports/research_maturity")
    return parser.parse_args()

def main():
    args = parse_args()
    candidates = read(args.candidate_audit, {
        "drug_id", "disease_id", "eligibility_status", "model_score",
    })
    trials = read(args.clinical_trials, {
        "drug_id", "disease_id", "clinical_evidence_state", "nct_id",
        "overall_status",
    }, optional=True)
    papers = read(args.pubmed, {
        "drug_id", "disease_id", "pmid", "publication_types",
    }, optional=True)
    adjudications = read(args.adjudications, {
        "drug_id", "disease_id", "clinical_outcome_verdict",
    }, optional=True)

    if not adjudications.empty:
        adjudications["clinical_outcome_verdict"] = adjudications[
            "clinical_outcome_verdict"
        ].map(upper)
        invalid = set(adjudications["clinical_outcome_verdict"]) - ALLOWED
        if invalid:
            raise ValueError(f"Unsupported clinical outcome verdicts: {sorted(invalid)}")
        if adjudications.duplicated(["drug_id", "disease_id"]).any():
            raise ValueError("Adjudications must contain one final verdict per pair.")
        adjudications = adjudications[[
            "drug_id", "disease_id", "clinical_outcome_verdict"
        ]]

    tracker = candidates.merge(
        summarise_trials(trials), on=["drug_id", "disease_id"], how="left"
    ).merge(
        summarise_papers(papers), on=["drug_id", "disease_id"], how="left"
    ).merge(
        adjudications, on=["drug_id", "disease_id"], how="left"
    )
    for column in ("registered_trial_count", "pubmed_record_count"):
        tracker[column] = tracker[column].fillna(0).astype(int)
    for column in ("needs_outcome_review", "trial_active", "literature_retrieved"):
        tracker[column] = tracker[column].fillna(False).astype(bool)
    for column in ("clinical_states", "trial_ids", "trial_statuses", "pubmed_ids",
                   "publication_types", "clinical_outcome_verdict"):
        tracker[column] = tracker[column].fillna("")

    stages = tracker.apply(stage, axis=1, result_type="expand")
    stages.columns = ["research_maturity_level", "research_maturity_stage",
                      "research_status", "recommended_next_action"]
    tracker = pd.concat([tracker, stages], axis=1).sort_values(
        ["research_maturity_level", "model_score"], ascending=[False, False]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "candidate_research_maturity_tracker.csv"
    tracker.to_csv(output, index=False)
    summary = {
        "input_pairs": int(len(tracker)),
        "mechanistically_compatible_pairs": int(
            tracker["eligibility_status"].eq(PASS).sum()
        ),
        "research_status_counts": tracker["research_status"].value_counts().to_dict(),
        "maturity_stage_counts": tracker["research_maturity_stage"].value_counts().to_dict(),
        "policy": {
            "retrieved_papers_and_trials": "research-progress signals, not efficacy claims",
            "human_benefit_or_harm": "requires a source-cited outcome adjudication",
            "negative_or_safety_outcomes": "excluded from hypothesis promotion",
        },
    }
    (args.output_dir / "research_maturity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote research maturity tracker to {output}")

if __name__ == "__main__":
    main()
