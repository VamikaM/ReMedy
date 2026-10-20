"""Build a conservative Alzheimer target review panel from persisted evidence tables.

The panel prioritizes targets for signature enrichment by public evidence coverage
and availability of phase-4 mechanism records. It does not call a target causal
or infer that a medicine benefits Alzheimer disease.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def joined(values: pd.Series, limit: int | None = None) -> str:
    unique = sorted({clean(value) for value in values if clean(value)})
    return " | ".join(unique[:limit] if limit else unique)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_path = args.curated_dir / "expanded_sources" / "open_targets" / "target_evidence_summary.csv"
    catalogue_path = args.curated_dir / "expanded_sources" / "chembl" / "approved_drug_target_catalog.csv"
    molecules_path = args.curated_dir / "expanded_sources" / "chembl" / "approved_molecule_registry.csv"

    evidence = pd.read_csv(evidence_path, dtype=str).fillna("")
    catalogue = pd.read_csv(catalogue_path, dtype=str).fillna("")
    molecules = pd.read_csv(molecules_path, dtype=str).fillna("")
    for frame in (evidence, catalogue, molecules):
        for column in frame.columns:
            frame[column] = frame[column].map(clean)

    molecule_name = dict(zip(molecules.get("molecule_chembl_id", []), molecules.get("pref_name", [])))
    catalogue["drug_name"] = catalogue["drug_id"].map(molecule_name).fillna("")
    coverage = catalogue.groupby("protein_id", as_index=False).agg(
        phase4_drug_count=("drug_id", "nunique"),
        phase4_drug_names=("drug_name", lambda values: joined(values, limit=30)),
        phase4_drug_actions=("action_type", joined),
        direct_mechanism_record_count=("evidence_id", "nunique"),
    )
    panel = evidence.merge(coverage, on="protein_id", how="left").fillna("")
    numeric = ["datasource_count", "evidence_record_count", "phase4_drug_count", "direct_mechanism_record_count"]
    for column in numeric:
        panel[column] = pd.to_numeric(panel.get(column), errors="coerce").fillna(0).astype(int)

    panel["target_review_tier"] = "T4_NO_DRUG_CATALOGUE"
    panel.loc[
        panel["phase4_drug_count"].gt(0)
        & panel["evidence_coverage_category"].eq("SINGLE_DATASOURCE_CONTEXT"),
        "target_review_tier",
    ] = "T3_SINGLE_SOURCE_DRUGGABLE_CONTEXT"
    panel.loc[
        panel["phase4_drug_count"].gt(0)
        & panel["evidence_coverage_category"].eq("MULTI_DATASOURCE_CONTEXT"),
        "target_review_tier",
    ] = "T2_MULTI_SOURCE_DRUGGABLE_CONTEXT"
    panel.loc[
        panel["phase4_drug_count"].gt(0)
        & panel["evidence_coverage_category"].eq("MULTI_DATASOURCE_CONTEXT")
        & panel["datasource_count"].ge(3),
        "target_review_tier",
    ] = "T1_PRIORITY_FOR_SIGNATURE_ENRICHMENT"

    panel["next_evidence_action"] = "No phase-4 mechanism catalogue: retain as disease-context target only."
    panel.loc[
        panel["target_review_tier"].eq("T3_SINGLE_SOURCE_DRUGGABLE_CONTEXT"),
        "next_evidence_action",
    ] = "Verify disease relevance in independent literature before drug signature enrichment."
    panel.loc[
        panel["target_review_tier"].eq("T2_MULTI_SOURCE_DRUGGABLE_CONTEXT"),
        "next_evidence_action",
    ] = "Collect disease and drug expression signatures; assess pathway and exposure context."
    panel.loc[
        panel["target_review_tier"].eq("T1_PRIORITY_FOR_SIGNATURE_ENRICHMENT"),
        "next_evidence_action",
    ] = "Prioritize for disease-signature reversal and pathway-convergence analysis."
    panel["interpretation_warning"] = (
        "Tier reflects available source coverage and druggability only; it is not a causal "
        "target verdict or an efficacy ranking."
    )

    panel = panel.sort_values(
        ["target_review_tier", "datasource_count", "evidence_record_count", "phase4_drug_count", "protein_label"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    out_dir = args.curated_dir / "target_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / "target_evidence_panel.csv", index=False)
    priority = panel.loc[
        panel["target_review_tier"].eq("T1_PRIORITY_FOR_SIGNATURE_ENRICHMENT")
    ].copy()
    priority.to_csv(out_dir / "signature_enrichment_target_queue.csv", index=False)

    summary = {
        "panel_proteins": int(len(panel)),
        "tier_counts": {str(key): int(value) for key, value in panel["target_review_tier"].value_counts().items()},
        "signature_enrichment_priority_targets": int(len(priority)),
        "policy": (
            "Targets enter signature enrichment based on evidence-context coverage and "
            "source-backed phase-4 mechanism availability. Human review remains required "
            "before a causal or therapeutic interpretation."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote conservative Alzheimer target panel to {out_dir}")


if __name__ == "__main__":
    main()
