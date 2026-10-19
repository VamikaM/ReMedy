"""Create ReMedy V2.1's diverse, evidence-traceable research shortlist.

This is a post-model decision layer for the frozen Evidence-Aware R-GCN (V2).
It does not retrain or alter model metrics. It:
1. retains only candidates classified as novel hypotheses requiring validation,
2. separates mechanistic hypotheses from literature-context items,
3. removes near-duplicate mechanisms per disease, and
4. writes an auditable next-action field for each retained or withheld candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--novelty-ranking",
        type=Path,
        default=root / "reports" / "novelty_ranking" / "novelty_aware_hypothesis_ranking.csv",
    )
    parser.add_argument(
        "--training-graph",
        type=Path,
        default=root / "data" / "model" / "training_graph_edges.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports" / "final_v21",
    )
    parser.add_argument("--top-per-disease", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.80)
    return parser.parse_args()


def mechanism_overlap(left: set[str], right: set[str]) -> float:
    """Overlap coefficient: detects when one supported mechanism is a subset of another."""
    smallest = min(len(left), len(right))
    return len(left & right) / smallest if smallest else 0.0


def compatible_protein_set(value: object) -> set[str]:
    return {
        item.strip()
        for item in str(value).split("|")
        if item and item.strip() and str(item).lower() != "nan"
    }


def next_action(stage: str) -> str:
    actions = {
        "M1_MECHANISTIC_HYPOTHESIS":
            "Review the mechanism and retrieve independent disease-specific literature.",
        "M2_LITERATURE_CONTEXT_RETRIEVED":
            "Screen retrieved articles for direct efficacy, harm, or non-relevant context.",
        "M3_HUMAN_RESEARCH_IDENTIFIED":
            "Adjudicate trial outcomes before any hypothesis promotion.",
        "M4_OUTCOME_ADJUDICATED":
            "Use the cited outcome verdict; do not promote without a positive source-backed result.",
    }
    return actions.get(stage, "Complete evidence review before further prioritisation.")


def main() -> None:
    args = parse_args()
    ranking = pd.read_csv(args.novelty_ranking, dtype={"drug_id": str, "disease_id": str})
    graph = pd.read_parquet(args.training_graph)
    graph["source_id"] = graph["source_id"].astype(str)
    graph["target_id"] = graph["target_id"].astype(str)

    required = {
        "drug_id", "disease_id", "disease_name", "drug_name",
        "novelty_priority_score", "candidate_class",
        "research_maturity_stage", "research_status",
    }
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Novelty ranking is missing columns: {sorted(missing)}")

    # Full drug target profiles are often different even when the same
    # disease-relevant mechanism is supported. Diversity must therefore use
    # the compatible proteins passed by the evidence gate.
    if "compatible_proteins" not in ranking.columns:
        raise ValueError("Novelty ranking must include compatible_proteins.")
    ranking["compatible_protein_profile"] = ranking["compatible_proteins"].map(
        compatible_protein_set
    )
    ranking["compatible_protein_count"] = ranking[
        "compatible_protein_profile"
    ].map(len)
    ranking["v21_next_action"] = ranking["research_maturity_stage"].map(next_action)

    eligible = ranking.loc[
        ranking["candidate_class"].eq("NOVEL_HYPOTHESIS_REQUIRING_VALIDATION")
    ].copy()
    eligible = eligible.sort_values(
        ["disease_id", "novelty_priority_score", "drug_id"],
        ascending=[True, False, True],
    )

    reviewed_groups = []
    for disease_id, group in eligible.groupby("disease_id", sort=False):
        selected: list[tuple[str, set[str]]] = []
        decisions = []
        for row in group.itertuples(index=False):
            profile = row.compatible_protein_profile
            max_similarity = max(
                (mechanism_overlap(profile, selected_profile)
                 for _, selected_profile in selected),
                default=0.0,
            )
            if not profile:
                decision = "WITHHELD_NO_SUPPORTED_MECHANISM"
            elif len(selected) >= args.top_per_disease:
                decision = "WITHHELD_DISEASE_QUOTA"
            elif max_similarity >= args.similarity_threshold:
                decision = "WITHHELD_MECHANISTICALLY_REDUNDANT"
            else:
                decision = "RETAINED_RESEARCH_HYPOTHESIS"
                selected.append((row.drug_id, profile))
            decisions.append((row.drug_id, decision, max_similarity))
        decision_frame = pd.DataFrame(
            decisions,
            columns=["drug_id", "v21_decision", "max_similarity_to_retained"],
        )
        reviewed_groups.append(group.merge(decision_frame, on="drug_id", how="left"))

    final_audit = (
        pd.concat(reviewed_groups, ignore_index=True)
        if reviewed_groups else eligible.assign(
            v21_decision=pd.Series(dtype=str),
            max_similarity_to_retained=pd.Series(dtype=float),
        )
    )
    retained = final_audit.loc[
        final_audit["v21_decision"].eq("RETAINED_RESEARCH_HYPOTHESIS")
    ].copy()
    retained.insert(
        0,
        "research_priority_rank",
        retained.groupby("disease_id").cumcount() + 1,
    )

    # Literature-context items remain visible for audit, but are not promoted
    # as novel hypotheses until a reviewer establishes direct relevance.
    context_queue = ranking.loc[
        ranking["research_maturity_stage"].eq("M2_LITERATURE_CONTEXT_RETRIEVED")
    ].copy().sort_values(
        ["disease_id", "novelty_priority_score", "drug_id"],
        ascending=[True, False, True],
    )
    context_queue["v21_next_action"] = context_queue[
        "research_maturity_stage"
    ].map(next_action)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_audit.to_csv(args.output_dir / "v21_hypothesis_decision_audit.csv", index=False)
    retained.to_csv(args.output_dir / "v21_retained_research_hypotheses.csv", index=False)
    context_queue.to_csv(args.output_dir / "v21_literature_relevance_queue.csv", index=False)

    summary = {
        "input_mechanistically_compatible_pairs": int(len(ranking)),
        "novel_hypothesis_candidates_before_diversification": int(len(eligible)),
        "retained_diverse_research_hypotheses": int(len(retained)),
        "withheld_mechanistically_redundant": int(
            final_audit["v21_decision"].eq("WITHHELD_MECHANISTICALLY_REDUNDANT").sum()
        ) if not final_audit.empty else 0,
        "withheld_disease_quota": int(
            final_audit["v21_decision"].eq("WITHHELD_DISEASE_QUOTA").sum()
        ) if not final_audit.empty else 0,
        "literature_relevance_queue": int(len(context_queue)),
        "policy": {
            "ranking_model": "Frozen Evidence-Aware R-GCN V2",
            "post_model_ranking": "specificity/pathway/class-aware novelty score",
            "diversification": "At most three non-near-duplicate evidence-supported protein mechanisms per disease",
            "promotion": "Only M1/M2 candidates with the prior evidence gate passed; source review remains required",
        },
    }
    (args.output_dir / "v21_shortlist_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote final V2.1 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
