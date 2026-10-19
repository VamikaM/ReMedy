"""Evaluate manually curated ReMedy evidence chains.

Only source-cited verdicts can advance a graph-generated candidate beyond the
initial mechanism/context screen. This script creates an auditable status and
next action; it does not infer missing biological or clinical evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalized(value: object) -> str:
    return clean(value).upper().replace(" ", "_").replace("-", "_")


def has_citation(value: object) -> bool:
    return bool(clean(value))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_dir = root / "data" / "curation" / "evidence_chain_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_dir)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "evidence_chain" / "candidate_evidence_chain_status.csv",
    )
    return parser.parse_args()


def stage_for(target: str, exposure: str, phenotype: str, human: str, safety: str) -> tuple[str, str]:
    if safety == "EXCLUDE":
        return "EXCLUDED_SAFETY_EVIDENCE", "Do not promote; retain the cited safety evidence."
    if human == "NEGATIVE":
        return "EXCLUDED_NEGATIVE_HUMAN_EVIDENCE", "Do not promote; retain the cited negative human outcome."
    if target != "SUPPORTED":
        return "WITHHELD_TARGET_CAUSALITY_UNRESOLVED", "Establish target role and desired modulation with a source."
    if exposure != "SUPPORTED":
        return "WITHHELD_DISEASE_RELEVANT_EXPOSURE_UNRESOLVED", "Establish disease-relevant tissue exposure with a source."
    if phenotype == "NEGATIVE":
        return "WITHHELD_NEGATIVE_PRECLINICAL_EVIDENCE", "Review the cited negative phenotype evidence."
    if human == "POSITIVE":
        return "HUMAN_EVIDENCE_IDENTIFIED", "Review study quality, population, endpoint, and benefit-risk evidence."
    if human == "TRIAL_OUTCOME_UNRESOLVED":
        return "HUMAN_OUTCOME_ADJUDICATION_REQUIRED", "Review the cited primary study outcome before promotion."
    if phenotype == "POSITIVE":
        return "PRECLINICAL_REPURPOSING_CANDIDATE", "Plan independent validation in a disease-relevant model."
    if human == "NO_HUMAN_EVIDENCE":
        return "MECHANISTIC_HYPOTHESIS_PRECLINICAL_EVIDENCE_REQUIRED", "Seek or generate disease-relevant phenotype evidence."
    return "WITHHELD_PHENOTYPE_OR_HUMAN_EVIDENCE_UNRESOLVED", "Complete phenotype and human-evidence review."


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.workspace / "candidate_evidence_chain.csv", dtype=str).fillna("")
    targets = pd.read_csv(args.workspace / "disease_target_review.csv", dtype=str).fillna("")
    exposures = pd.read_csv(args.workspace / "drug_exposure_review.csv", dtype=str).fillna("")
    outcomes = pd.read_csv(
        args.workspace / "phenotype_human_outcome_review.csv", dtype=str
    ).fillna("")

    for frame in (candidates, targets, exposures, outcomes):
        if "candidate_id" in frame.columns:
            frame["candidate_id"] = frame["candidate_id"].map(clean)

    targets["target_review_verdict"] = targets["target_review_verdict"].map(normalized)
    targets["citation_present"] = targets["source_citation"].map(has_citation)
    targets["target_role"] = targets["target_role"].map(normalized)
    targets["desired_modulation"] = targets["desired_modulation"].map(normalized)
    valid_targets = targets.loc[
        targets["target_review_verdict"].eq("SUPPORTED")
        & targets["citation_present"]
        & targets["target_role"].isin({"CAUSAL_DRIVER", "PROTECTIVE_FACTOR"})
        & ~targets["desired_modulation"].isin({"", "UNREVIEWED", "UNCERTAIN"})
    ]
    target_support = (
        valid_targets.groupby("disease_id")["protein_id"]
        .agg(lambda values: set(values.map(clean)))
        .to_dict()
    )

    exposures["exposure_review_verdict"] = exposures["exposure_review_verdict"].map(normalized)
    exposures["citation_present"] = exposures["source_citation"].map(has_citation)
    exposure_support = set(
        exposures.loc[
            exposures["exposure_review_verdict"].eq("SUPPORTED")
            & exposures["citation_present"], "candidate_id"
        ]
    )

    outcomes["preclinical_phenotype_verdict"] = outcomes[
        "preclinical_phenotype_verdict"
    ].map(normalized)
    outcomes["human_outcome_verdict"] = outcomes["human_outcome_verdict"].map(normalized)
    outcomes["disease_relevant_safety_verdict"] = outcomes[
        "disease_relevant_safety_verdict"
    ].map(normalized)
    outcomes["citation_present"] = outcomes["source_citation"].map(has_citation)
    outcome_by_id = outcomes.set_index("candidate_id").to_dict("index")

    result = candidates.copy()
    stages, actions, target_flags = [], [], []
    for row in result.itertuples(index=False):
        supported_proteins = target_support.get(row.disease_id, set())
        candidate_proteins = {
            item.strip() for item in clean(row.compatible_proteins).split("|")
            if item.strip()
        }
        target = "SUPPORTED" if supported_proteins & candidate_proteins else "PENDING"
        exposure = "SUPPORTED" if row.candidate_id in exposure_support else "PENDING"
        outcome = outcome_by_id.get(row.candidate_id, {})
        citation_present = bool(outcome.get("citation_present", False))
        phenotype = outcome.get("preclinical_phenotype_verdict", "PENDING")
        human = outcome.get("human_outcome_verdict", "PENDING")
        safety = outcome.get("disease_relevant_safety_verdict", "PENDING")
        # Outcome claims without a source are treated as unresolved, never promoted.
        if not citation_present:
            if phenotype != "PENDING":
                phenotype = "PENDING"
            if human != "PENDING":
                human = "PENDING"
            if safety != "PENDING":
                safety = "PENDING"
        stage, action = stage_for(target, exposure, phenotype, human, safety)
        stages.append(stage)
        actions.append(action)
        target_flags.append(target)

    result["target_gate_status"] = target_flags
    result["exposure_gate_status"] = [
        "SUPPORTED" if candidate_id in exposure_support else "PENDING"
        for candidate_id in result["candidate_id"]
    ]
    result["evidence_chain_stage"] = stages
    result["next_required_action"] = actions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "input_candidates": int(len(result)),
        "stage_counts": result["evidence_chain_stage"].value_counts().to_dict(),
        "promotion_policy": "Source-cited target and exposure support are required before phenotype or human evidence stages.",
    }
    summary_path = args.output.with_name("candidate_evidence_chain_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote evidence-chain status file to {args.output}")


if __name__ == "__main__":
    main()
