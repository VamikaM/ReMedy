"""Audit the Alzheimer disease V1 candidate pool before enrichment or model training.

This report identifies whether the pool is dominated by broad, weak one-target
matches. It does not score therapeutic suitability or exclude candidates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports" / "data_quality" / "alzheimers_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = pd.read_csv(
        args.curated_dir / "evidence_labels" / "candidate_pool.csv", dtype=str
    ).fillna("")
    targets = pd.read_csv(
        args.curated_dir / "entities" / "disease_target_profile.csv", dtype=str
    ).fillna("")
    mechanisms = pd.read_csv(
        args.curated_dir / "relations" / "drug_target_mechanisms.csv", dtype=str
    ).fillna("")

    pool["matched_target_count"] = pd.to_numeric(
        pool["matched_target_count"], errors="coerce"
    ).fillna(0).astype(int)
    target_coverage = (
        mechanisms.groupby(["protein_id", "protein_label"], as_index=False)
        .agg(
            matched_drug_count=("drug_id", "nunique"),
            actions=("action_type", lambda values: " | ".join(sorted(set(values)))),
            drug_names=("drug_name", lambda values: " | ".join(sorted(set(values)))),
        )
        .sort_values(["matched_drug_count", "protein_label"], ascending=[False, True])
    )
    target_coverage = targets.merge(
        target_coverage, on=["protein_id", "protein_label"], how="left"
    ).fillna("")
    target_coverage["matched_drug_count"] = pd.to_numeric(
        target_coverage["matched_drug_count"], errors="coerce"
    ).fillna(0).astype(int)

    status_counts = pool["candidate_status"].value_counts().to_dict()
    target_match_counts = (
        pool["matched_target_count"].value_counts().sort_index().astype(int).to_dict()
    )
    controls = pool.loc[
        pool["candidate_status"].eq("KNOWN_USE_CONTROL_NOT_NOVEL")
        | pool["candidate_status"].eq("CONTRAINDICATION_CONTROL_REVIEW")
    ].copy()

    summary = {
        "candidate_pool_drugs": int(len(pool)),
        "disease_targets": int(len(targets)),
        "drug_target_mechanism_rows": int(len(mechanisms)),
        "candidate_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "matched_target_count_distribution": {
            str(key): int(value) for key, value in target_match_counts.items()
        },
        "one_target_candidate_fraction": round(
            float((pool["matched_target_count"] == 1).mean()), 4
        ) if len(pool) else 0.0,
        "known_or_contraindication_controls_in_pool": int(len(controls)),
        "highest_fanout_target": (
            target_coverage.iloc[0][["protein_id", "protein_label", "matched_drug_count"]].to_dict()
            if len(target_coverage) else {}
        ),
        "interpretation": (
            "This is a data-quality audit. A drug that shares one disease-linked target "
            "is only a retrieval candidate. Enrichment with disease expression, exposure, "
            "known-use, trial outcome, and safety evidence is required before modelling."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_coverage.to_csv(args.output_dir / "target_candidate_coverage.csv", index=False)
    pool.sort_values(
        ["matched_target_count", "drug_name"], ascending=[False, True]
    ).to_csv(args.output_dir / "candidate_pool_by_target_coverage.csv", index=False)
    controls.to_csv(args.output_dir / "known_relation_controls.csv", index=False)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote Alzheimer V1 data-quality audit to {args.output_dir}")


if __name__ == "__main__":
    main()
