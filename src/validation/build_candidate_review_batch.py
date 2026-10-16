"""Build a bounded, reproducible batch for ReMedy clinical evidence retrieval.

The graph may score many unknown pairs, but evidence retrieval is deliberately
performed only for the top mechanism-consistent pairs per disease.  This keeps
external-source matching focused, auditable, and computationally practical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path, required=True,
                        help="candidate_gate_audit.csv from evidence_gate.py")
    parser.add_argument("--top-k-per-disease", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/clinical_review_candidates.csv")
    parser.add_argument("--metadata", type=Path,
                        default=root / "reports/evidence_gate/clinical_review_batch_metadata.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k_per_disease < 1:
        raise ValueError("--top-k-per-disease must be at least 1")
    audit = pd.read_csv(args.candidate_audit, dtype={"drug_id": str, "disease_id": str})
    required = {"drug_id", "disease_id", "eligibility_status", "model_score"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"{args.candidate_audit.name} is missing columns: {sorted(missing)}")

    candidates = audit.loc[audit["eligibility_status"].eq("EVIDENCE_SUPPORTED")].copy()
    candidates = candidates.sort_values(
        ["disease_id", "model_score", "drug_id"],
        ascending=[True, False, True],
    )
    candidates["clinical_retrieval_rank"] = (
        candidates.groupby("disease_id").cumcount() + 1
    )
    batch = candidates.loc[
        candidates["clinical_retrieval_rank"].le(args.top_k_per_disease)
    ].copy()
    batch = batch.sort_values(
        ["disease_id", "clinical_retrieval_rank", "drug_id"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(args.output, index=False)

    metadata = {
        "input_audit_pairs": int(len(audit)),
        "mechanism_consistent_pairs": int(len(candidates)),
        "top_k_per_disease": args.top_k_per_disease,
        "retrieval_batch_pairs": int(len(batch)),
        "disease_counts": batch.groupby("disease_id").size().to_dict(),
        "selection_policy": (
            "Top model-score pairs that passed the prior mechanism/evidence gate; "
            "external clinical retrieval is run only on this bounded batch."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote clinical evidence retrieval batch to {args.output}")


if __name__ == "__main__":
    main()
