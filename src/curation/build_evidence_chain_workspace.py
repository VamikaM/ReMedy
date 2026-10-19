"""Create ReMedy's evidence-chain curation workspace.

The graph model supplies hypotheses; this script turns the hypotheses that passed
the initial mechanism/context screen into source-citable review tables. No row
is promoted by this script. A reviewer must establish causal target role,
disease-relevant exposure, phenotype evidence, and human outcome where present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def unique_join(values: pd.Series) -> str:
    return " | ".join(sorted({clean(value) for value in values if clean(value)}))


def split_pipe(value: object) -> list[str]:
    return [item.strip() for item in clean(value).split("|") if item.strip()]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path,
        default=root / "reports" / "candidate_eligibility" / "candidate_gate_audit.csv")
    parser.add_argument("--disease-effects", type=Path,
        default=root / "data" / "processed" / "disease_target_effects.csv")
    parser.add_argument("--output-dir", type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.candidate_audit, dtype={"drug_id": str, "disease_id": str})
    effects = pd.read_csv(args.disease_effects, dtype={"disease_id": str, "protein_id": str})

    required_audit = {
        "drug_id", "drug_name", "disease_id", "disease_name", "model_score",
        "eligibility_status", "compatible_proteins", "compatible_drug_actions",
        "compatible_disease_effects", "mechanism_sources",
        "mechanism_evidence_ids", "disease_effect_sources",
        "disease_effect_evidence_ids", "safety_status",
    }
    missing = required_audit - set(audit.columns)
    if missing:
        raise ValueError(f"Candidate audit missing columns: {sorted(missing)}")
    required_effects = {
        "disease_id", "protein_id", "disease_effect", "evidence_score",
        "source", "evidence_id",
    }
    missing = required_effects - set(effects.columns)
    if missing:
        raise ValueError(f"Disease effects missing columns: {sorted(missing)}")

    for frame, columns in (
        (audit, ["drug_id", "disease_id"]),
        (effects, ["disease_id", "protein_id"]),
    ):
        for column in columns:
            frame[column] = frame[column].map(clean)

    candidates = audit.loc[audit["eligibility_status"].eq("EVIDENCE_SUPPORTED")].copy()
    if candidates.empty:
        raise ValueError("No EVIDENCE_SUPPORTED rows found; run evidence_gate.py first.")
    candidates = candidates.sort_values(
        ["disease_id", "model_score", "drug_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    candidates.insert(0, "candidate_id", [
        f"REM-{index + 1:03d}" for index in range(len(candidates))
    ])

    target_rows = []
    for row in candidates.itertuples(index=False):
        for protein_id in split_pipe(row.compatible_proteins):
            target_rows.append({
                "disease_id": row.disease_id,
                "disease_name": row.disease_name,
                "protein_id": protein_id,
            })
    candidate_targets = pd.DataFrame(target_rows).drop_duplicates()
    effects["evidence_score"] = pd.to_numeric(
        effects["evidence_score"], errors="coerce"
    ).fillna(0.0)
    effect_summary = effects.groupby(
        ["disease_id", "protein_id"], as_index=False
    ).agg(
        raw_disease_effect=("disease_effect", unique_join),
        open_targets_evidence_score_max=("evidence_score", "max"),
        raw_effect_source=("source", unique_join),
        raw_effect_evidence_id=("evidence_id", unique_join),
    )
    target_review = candidate_targets.merge(
        effect_summary, on=["disease_id", "protein_id"], how="left"
    ).sort_values(["disease_name", "protein_id"]).reset_index(drop=True)
    target_review["target_role"] = "UNREVIEWED"
    target_review["desired_modulation"] = "UNREVIEWED"
    target_review["relevant_tissue_or_cell"] = "UNREVIEWED"
    target_review["causal_evidence_tier"] = "UNREVIEWED"
    target_review["target_review_verdict"] = "PENDING"
    target_review["source_citation"] = ""
    target_review["review_notes"] = ""

    pair_columns = [
        "candidate_id", "drug_id", "drug_name", "disease_id", "disease_name",
        "model_score", "compatible_proteins", "compatible_drug_actions",
        "compatible_disease_effects", "mechanism_sources",
        "mechanism_evidence_ids", "disease_effect_sources",
        "disease_effect_evidence_ids", "safety_status",
    ]
    candidate_chain = candidates[pair_columns].copy()
    candidate_chain["initial_model_stage"] = "MECHANISM_AND_CONTEXT_SCREEN_PASSED"
    candidate_chain["target_gate_status"] = "PENDING"
    candidate_chain["exposure_gate_status"] = "PENDING"
    candidate_chain["phenotype_evidence_status"] = "PENDING"
    candidate_chain["human_evidence_status"] = "PENDING"
    candidate_chain["safety_review_status"] = "PENDING"
    candidate_chain["evidence_chain_stage"] = "WITHHELD_PENDING_CURATION"
    candidate_chain["next_required_action"] = (
        "Complete target, exposure, phenotype, human-outcome and safety review."
    )
    candidate_chain["reviewer_citation"] = ""
    candidate_chain["review_notes"] = ""

    exposure_review = candidate_chain[
        ["candidate_id", "drug_id", "drug_name", "disease_id", "disease_name",
         "compatible_proteins", "compatible_drug_actions"]
    ].copy()
    exposure_review["relevant_tissue_or_cell"] = "UNREVIEWED"
    exposure_review["disease_relevant_exposure"] = "UNREVIEWED"
    exposure_review["exposure_evidence_type"] = "UNREVIEWED"
    exposure_review["exposure_review_verdict"] = "PENDING"
    exposure_review["source_citation"] = ""
    exposure_review["review_notes"] = ""

    outcome_review = candidate_chain[
        ["candidate_id", "drug_id", "drug_name", "disease_id", "disease_name"]
    ].copy()
    outcome_review["preclinical_phenotype_verdict"] = "PENDING"
    outcome_review["human_outcome_verdict"] = "PENDING"
    outcome_review["disease_relevant_safety_verdict"] = "PENDING"
    outcome_review["source_citation"] = ""
    outcome_review["review_notes"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_chain.to_csv(args.output_dir / "candidate_evidence_chain.csv", index=False)
    target_review.to_csv(args.output_dir / "disease_target_review.csv", index=False)
    exposure_review.to_csv(args.output_dir / "drug_exposure_review.csv", index=False)
    outcome_review.to_csv(args.output_dir / "phenotype_human_outcome_review.csv", index=False)

    readme = """# ReMedy Evidence-Chain Curation Workspace

1. disease_target_review.csv: establish whether each protein is causal,
   protective, biomarker, or unsupported; enter desired modulation, tissue/cell,
   evidence tier, citation and verdict.
2. drug_exposure_review.csv: establish whether a drug reaches the relevant
   tissue/cell at disease-relevant exposure; record a citation and verdict.
3. phenotype_human_outcome_review.csv: record phenotype, human outcome, and
   disease-specific safety verdicts with citations.
4. candidate_evidence_chain.csv: generated audit table. Do not manually promote
   candidates here; use the evaluator after review.

Controlled vocabularies:
target_review_verdict: PENDING | SUPPORTED | UNSUPPORTED
exposure_review_verdict: PENDING | SUPPORTED | UNSUPPORTED
preclinical_phenotype_verdict: PENDING | POSITIVE | NEGATIVE | NOT_FOUND
human_outcome_verdict: PENDING | POSITIVE | NEGATIVE | TRIAL_OUTCOME_UNRESOLVED | NO_HUMAN_EVIDENCE
disease_relevant_safety_verdict: PENDING | CLEAR_FOR_RESEARCH | EXCLUDE

A supported or negative verdict must include a direct source citation.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "screened_hypotheses": int(len(candidate_chain)),
        "unique_disease_protein_targets_for_review": int(len(target_review)),
        "diseases_in_scope": int(candidate_chain["disease_id"].nunique()),
        "policy": "No candidate is promoted before source-cited review completes every evidence gate.",
    }
    (args.output_dir / "workspace_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote curation workspace to {args.output_dir}")


if __name__ == "__main__":
    main()
