"""Recreate ReMedy's V2 evidence package from a fresh Kaggle runtime.

This orchestrator is deliberately explicit and reproducible. It rebuilds derived
evidence tables, trains the Evidence-Aware R-GCN, scores unknown pairs, applies
the evidence gate, retrieves bounded external evidence, and writes the maturity,
novelty, review, and V3-label preparation outputs.

Prerequisites:
* Run from the repository root.
* Internet enabled for ChEMBL, Open Targets, openFDA, ClinicalTrials.gov, and PubMed.
* torch_geometric installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    command = [sys.executable, *parts]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("src/preprocessing/fetch_chembl_mechanisms.py")
    run("src/preprocessing/fetch_open_targets_direction_evidence.py")
    run(
        "src/modeling/train_evidence_aware_rgcn.py",
        "--seeds", "7,13,29,42,73",
        "--max-epochs", "150",
        "--patience", "20",
    )
    run("src/modeling/generate_evidence_aware_candidate_rankings.py")
    run(
        "src/preprocessing/fetch_openfda_context_screen.py",
        "--ranking",
        "reports/model_evaluation/evidence_aware_rgcn_raw_internal_scores.csv",
    )
    run(
        "src/validation/evidence_gate.py",
        "--ranking",
        "reports/model_evaluation/evidence_aware_rgcn_raw_internal_scores.csv",
        "--nodes", "data/processed/nodes.parquet",
        "--edges", "data/processed/edges.parquet",
        "--mechanisms", "data/processed/drug_mechanisms.csv",
        "--disease-effects", "data/processed/disease_target_effects.csv",
        "--safety", "data/processed/drug_safety_context.csv",
    )
    run(
        "src/validation/build_candidate_review_batch.py",
        "--candidate-audit",
        "reports/candidate_eligibility/candidate_gate_audit.csv",
        "--top-k-per-disease", "50",
    )
    run(
        "src/preprocessing/fetch_clinical_trial_evidence.py",
        "--pairs", "data/processed/clinical_review_candidates.csv",
    )
    run(
        "src/preprocessing/fetch_pubmed_evidence.py",
        "--pairs", "data/processed/clinical_review_candidates.csv",
        "--max-articles-per-pair", "10",
    )
    run(
        "src/validation/build_research_maturity_tracker.py",
        "--candidate-audit",
        "reports/candidate_eligibility/candidate_gate_audit.csv",
        "--clinical-trials", "data/processed/clinical_trial_evidence.csv",
        "--pubmed", "data/processed/pubmed_evidence.csv",
        "--retrieval-batch", "data/processed/clinical_review_candidates.csv",
    )
    run(
        "src/validation/rank_novelty_hypotheses.py",
        "--maturity",
        "reports/research_maturity/candidate_research_maturity_tracker.csv",
        "--raw-ranking",
        "reports/model_evaluation/evidence_aware_rgcn_raw_internal_scores.csv",
    )
    run(
        "src/validation/build_candidate_adjudication_template.py",
        "--candidate-audit",
        "reports/candidate_eligibility/candidate_gate_audit.csv",
    )
    run("src/validation/build_pu_label_table.py")
    print("\nReMedy V2 evidence package rebuilt successfully.")


if __name__ == "__main__":
    main()
