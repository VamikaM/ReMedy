"""Run ReMedy's conservative evidence-gated candidate workflow end to end.

Prerequisite: models/hybrid_rgcn.pt must exist from a completed Hybrid R-GCN
training run. The final researcher-facing output is written only after the
evidence gate succeeds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run("src/modeling/generate_hybrid_candidate_rankings.py")
    run("src/preprocessing/fetch_chembl_mechanisms.py")
    run("src/preprocessing/fetch_open_targets_direction_evidence.py")
    run("src/preprocessing/fetch_openfda_context_screen.py")
    run(
        "src/validation/evidence_gate.py",
        "--ranking", "reports/model_evaluation/hybrid_rgcn_raw_internal_scores.csv",
        "--nodes", "data/processed/nodes.parquet",
        "--edges", "data/processed/edges.parquet",
        "--mechanisms", "data/processed/drug_mechanisms.csv",
        "--disease-effects", "data/processed/disease_target_effects.csv",
        "--safety", "data/processed/drug_safety_context.csv",
    )
    print("\nEvidence-gated workflow completed.")
    print("Open reports/candidate_eligibility/evidence_supported_candidates.csv")
    print("for the default researcher-facing shortlist.")


if __name__ == "__main__":
    main()
