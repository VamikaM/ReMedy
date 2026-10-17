"""Rank ReMedy hypotheses with disease-specificity penalties.

The Hybrid R-GCN score is retained as a structural signal. This module reduces
hub bias by down-weighting proteins targeted by many drugs and associated with
many diseases, and it keeps research maturity separate from novelty ranking.
Its output is an auditable research-priority list, never a treatment claim.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PASS = "EVIDENCE_SUPPORTED"
EXCLUDED = {
    "EXCLUDED_NEGATIVE_HUMAN_EVIDENCE",
    "EXCLUDED_DISEASE_RELEVANT_SAFETY_SIGNAL",
}

def clean(value):
    return "" if pd.isna(value) else str(value).strip()

def proteins(value):
    return [item.strip() for item in clean(value).split("|") if item.strip()]

def minmax(values):
    values = pd.Series(values, dtype=float)
    spread = values.max() - values.min()
    return pd.Series(np.zeros(len(values)), index=values.index) if spread == 0 else (
        (values - values.min()) / spread
    )

def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maturity", type=Path,
                        default=root / "reports/research_maturity/candidate_research_maturity_tracker.csv")
    parser.add_argument("--raw-ranking", type=Path,
                        default=root / "reports/model_evaluation/hybrid_rgcn_raw_internal_scores.csv")
    parser.add_argument("--training-graph", type=Path,
                        default=root / "data/model/training_graph_edges.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports/novelty_ranking")
    return parser.parse_args()

def main():
    args = parse_args()
    maturity = pd.read_csv(args.maturity, dtype={"drug_id": str, "disease_id": str})
    raw = pd.read_csv(args.raw_ranking, dtype={"drug_id": str, "disease_id": str})
    graph = pd.read_parquet(args.training_graph)
    for frame in (maturity, raw, graph):
        for column in ("drug_id", "disease_id", "source_id", "target_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    required = {
        "drug_id", "disease_id", "eligibility_status", "compatible_proteins",
        "research_maturity_stage", "research_status",
    }
    missing = required - set(maturity.columns)
    if missing:
        raise ValueError(f"Maturity tracker is missing columns: {sorted(missing)}")
    if {"drug_id", "disease_id", "hybrid_rgcn_score"} - set(raw.columns):
        raise ValueError("Raw ranking must contain drug_id, disease_id, hybrid_rgcn_score")

    target_degree = graph.loc[graph["relation"].eq("targets")].groupby(
        "target_id"
    )["source_id"].nunique().to_dict()
    disease_degree = graph.loc[graph["relation"].eq("associated_with")].groupby(
        "target_id"
    )["source_id"].nunique().to_dict()

    score_lookup = raw.set_index(["drug_id", "disease_id"])["hybrid_rgcn_score"]
    result = maturity.loc[maturity["eligibility_status"].eq(PASS)].copy()
    result["hybrid_rgcn_score"] = [
        score_lookup.get((row.drug_id, row.disease_id), np.nan)
        for row in result.itertuples(index=False)
    ]
    if result["hybrid_rgcn_score"].isna().any():
        raise ValueError("Some maturity pairs were not found in raw ranking.")

    raw["within_disease_model_percentile"] = raw.groupby("disease_id")[
        "hybrid_rgcn_score"
    ].rank(pct=True, method="average")
    percentile = raw.set_index(["drug_id", "disease_id"])[
        "within_disease_model_percentile"
    ]
    result["within_disease_model_percentile"] = [
        percentile[(row.drug_id, row.disease_id)]
        for row in result.itertuples(index=False)
    ]

    def specificity(value):
        scores = []
        for protein in proteins(value):
            drug_count = target_degree.get(protein, 0)
            disease_count = disease_degree.get(protein, 0)
            if drug_count and disease_count:
                scores.append(1 / math.log2(1 + drug_count * disease_count))
        return float(np.mean(scores)) if scores else 0.0

    result["protein_specificity_raw"] = result["compatible_proteins"].map(specificity)
    result["protein_specificity_score"] = minmax(result["protein_specificity_raw"])
    result["shared_pathway_count"] = result.get(
        "shared_pathway_count", pd.Series(0, index=result.index)
    ).fillna(0).astype(float)
    result["pathway_convergence_score"] = minmax(
        np.log1p(result["shared_pathway_count"])
    )

    maturity_multiplier = {
        "M1_MECHANISTIC_HYPOTHESIS": 1.00,
        "M2_LITERATURE_CONTEXT_RETRIEVED": 0.95,
        "M3_HUMAN_RESEARCH_IDENTIFIED": 0.60,
        "M4_OUTCOME_ADJUDICATED": 0.00,
    }
    result["maturity_multiplier"] = result["research_maturity_stage"].map(
        maturity_multiplier
    ).fillna(0.0)
    result["base_novelty_score"] = (
        0.45 * result["within_disease_model_percentile"]
        + 0.35 * result["protein_specificity_score"]
        + 0.20 * result["pathway_convergence_score"]
    )
    result["novelty_priority_score"] = (
        result["base_novelty_score"] * result["maturity_multiplier"]
    )
    result.loc[result["research_status"].isin(EXCLUDED), "novelty_priority_score"] = 0.0
    result["candidate_class"] = np.where(
        result["research_maturity_stage"].isin(
            ["M1_MECHANISTIC_HYPOTHESIS", "M2_LITERATURE_CONTEXT_RETRIEVED"]
        ),
        "NOVEL_HYPOTHESIS_REQUIRING_VALIDATION",
        "NOT_A_NOVEL_HYPOTHESIS_SHORTLIST_ITEM",
    )
    result = result.sort_values(
        ["disease_id", "novelty_priority_score", "drug_id"],
        ascending=[True, False, True],
    )
    result["disease_novelty_rank"] = result.groupby("disease_id").cumcount() + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "novelty_aware_hypothesis_ranking.csv"
    result.to_csv(output, index=False)
    shortlist = result.loc[
        result["candidate_class"].eq("NOVEL_HYPOTHESIS_REQUIRING_VALIDATION")
    ].copy()
    shortlist.to_csv(args.output_dir / "novel_hypothesis_shortlist.csv", index=False)
    summary = {
        "input_mechanistically_compatible_pairs": int(len(result)),
        "novel_hypotheses_requiring_validation": int(len(shortlist)),
        "score": {
            "model_signal": 0.45,
            "protein_specificity": 0.35,
            "pathway_convergence": 0.20,
            "maturity_adjustment": "penalises unresolved/established human research",
        },
        "safeguard": "Output is a research-priority list, not a clinical recommendation.",
    }
    (args.output_dir / "novelty_ranking_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote novelty-aware ranking to {output}")

if __name__ == "__main__":
    main()
