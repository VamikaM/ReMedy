"""Build ReMedy's Alzheimer disease V1 curated discovery dataset.

This builder creates a narrow, provenance-preserving disease dataset from the
existing ReMedy knowledge graph and source-backed mechanism/evidence tables.
It is intentionally a data-construction step: genetic association is retained
as context and is never converted into a therapeutic-direction or efficacy
label.

Outputs are written below data/curated/alzheimers_v1/ and can be regenerated
without editing the broad ReMedy graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KNOWN_RELATIONS = {"indication", "off-label use", "contraindication"}


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def joined(values: pd.Series) -> str:
    result: list[str] = []
    for raw in values:
        value = clean(raw)
        if value and value not in result:
            result.append(value)
    return " | ".join(result)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=root / "data" / "processed" / "nodes.parquet")
    parser.add_argument("--edges", type=Path, default=root / "data" / "processed" / "edges.parquet")
    parser.add_argument(
        "--mechanisms",
        type=Path,
        default=root / "data" / "processed" / "drug_mechanisms.csv",
    )
    parser.add_argument(
        "--genetic-context",
        type=Path,
        default=root / "data" / "processed" / "disease_target_effects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "curated" / "alzheimers_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = pd.read_parquet(args.nodes).copy()
    edges = pd.read_parquet(args.edges).copy()
    mechanisms = pd.read_csv(args.mechanisms, dtype=str).fillna("")

    required_node_columns = {"node_id", "node_type", "label"}
    required_edge_columns = {"source_id", "relation", "target_id"}
    required_mechanism_columns = {"drug_id", "protein_id", "action_type", "source", "evidence_id"}
    if missing := required_node_columns - set(nodes.columns):
        raise ValueError(f"nodes file missing columns: {sorted(missing)}")
    if missing := required_edge_columns - set(edges.columns):
        raise ValueError(f"edges file missing columns: {sorted(missing)}")
    if missing := required_mechanism_columns - set(mechanisms.columns):
        raise ValueError(f"mechanisms file missing columns: {sorted(missing)}")

    for frame, columns in (
        (nodes, ["node_id", "node_type", "label"]),
        (edges, ["source_id", "relation", "target_id"]),
        (mechanisms, ["drug_id", "protein_id", "action_type", "source", "evidence_id"]),
    ):
        for column in columns:
            frame[column] = frame[column].map(clean)

    nodes["node_type_normalized"] = nodes["node_type"].str.lower()
    node_type = nodes.set_index("node_id")["node_type_normalized"].to_dict()
    node_label = nodes.set_index("node_id")["label"].to_dict()

    alzheimer_rows = nodes.loc[
        nodes["label"].str.casefold().eq("alzheimer disease".casefold())
        & nodes["node_type_normalized"].eq("disease")
    ]
    if len(alzheimer_rows) != 1:
        raise ValueError(
            "Expected exactly one disease node labelled 'Alzheimer disease'; found "
            f"{len(alzheimer_rows)}."
        )
    disease_id = clean(alzheimer_rows.iloc[0]["node_id"])
    disease_name = clean(alzheimer_rows.iloc[0]["label"])

    edges["relation_normalized"] = edges["relation"].str.lower()
    # Support either orientation while preserving the original relation/provenance.
    direct_target_edges = edges.loc[
        edges["source_id"].eq(disease_id)
        & edges["target_id"].map(node_type).eq("protein")
    ].copy()
    reverse_target_edges = edges.loc[
        edges["target_id"].eq(disease_id)
        & edges["source_id"].map(node_type).eq("protein")
    ].copy()
    if not reverse_target_edges.empty:
        reverse_target_edges[["source_id", "target_id"]] = reverse_target_edges[
            ["target_id", "source_id"]
        ]
    target_edges = pd.concat([direct_target_edges, reverse_target_edges], ignore_index=True)
    target_edges = target_edges.rename(columns={"target_id": "protein_id"})
    if target_edges.empty:
        raise RuntimeError("No Alzheimer disease–protein edges were found in the broad graph.")

    target_profile = (
        target_edges.groupby("protein_id", as_index=False)
        .agg(
            graph_relations=("relation", joined),
            graph_edge_count=("relation", "size"),
        )
    )
    target_profile["disease_id"] = disease_id
    target_profile["disease_name"] = disease_name
    target_profile["protein_label"] = target_profile["protein_id"].map(node_label).fillna("")

    if args.genetic_context.exists():
        genetic = pd.read_csv(args.genetic_context, dtype=str).fillna("")
        for column in ("disease_id", "protein_id"):
            if column in genetic:
                genetic[column] = genetic[column].map(clean)
        genetic = genetic.loc[genetic["disease_id"].eq(disease_id)].copy()
        if not genetic.empty:
            genetic_summary = genetic.groupby("protein_id", as_index=False).agg(
                genetic_association_signals=("disease_effect", joined),
                direction_on_target=("direction_on_target", joined),
                direction_on_trait=("direction_on_trait", joined),
                genetic_sources=("source", joined),
                genetic_evidence_ids=("evidence_id", joined),
                genetic_evidence_score_max=(
                    "evidence_score",
                    lambda values: pd.to_numeric(values, errors="coerce").max(),
                ),
                open_targets_disease_id=("open_targets_disease_id", joined),
                open_targets_target_id=("open_targets_target_id", joined),
            )
            target_profile = target_profile.merge(genetic_summary, on="protein_id", how="left")
    else:
        genetic = pd.DataFrame()

    target_profile = target_profile.fillna("")
    target_profile["genetic_context_warning"] = (
        "Genetic association is evidence context only; it is not a therapeutic direction or efficacy claim."
    )
    target_profile = target_profile.sort_values(["protein_label", "protein_id"]).reset_index(drop=True)
    target_ids = set(target_profile["protein_id"])

    mechanism_profile = mechanisms.loc[mechanisms["protein_id"].isin(target_ids)].copy()
    mechanism_profile["drug_name"] = mechanism_profile["drug_id"].map(node_label).fillna("")
    mechanism_profile["protein_label"] = mechanism_profile["protein_id"].map(node_label).fillna("")
    mechanism_profile["disease_id"] = disease_id
    mechanism_profile["disease_name"] = disease_name
    mechanism_profile = mechanism_profile[
        [
            "drug_id", "drug_name", "protein_id", "protein_label", "disease_id",
            "disease_name", "action_type", "source", "evidence_id",
        ]
    ].drop_duplicates().sort_values(["drug_name", "protein_label", "action_type"])

    drug_ids = set(nodes.loc[nodes["node_type_normalized"].eq("drug"), "node_id"])
    known = edges.loc[edges["relation_normalized"].isin(KNOWN_RELATIONS)].copy()
    known_direct = known.loc[
        known["source_id"].isin(drug_ids) & known["target_id"].eq(disease_id)
    ].copy()
    reverse_known = known.loc[
        known["target_id"].isin(drug_ids) & known["source_id"].eq(disease_id)
    ].copy()
    if not reverse_known.empty:
        reverse_known[["source_id", "target_id"]] = reverse_known[["target_id", "source_id"]]
    known_direct = pd.concat([known_direct, reverse_known], ignore_index=True)
    known_direct = known_direct.rename(columns={"source_id": "drug_id"})
    known_labels = (
        known_direct.groupby("drug_id", as_index=False)
        .agg(existing_relation=("relation", joined))
        if not known_direct.empty
        else pd.DataFrame(columns=["drug_id", "existing_relation"])
    )
    known_labels["drug_name"] = known_labels["drug_id"].map(node_label).fillna("")
    known_labels["disease_id"] = disease_id
    known_labels["disease_name"] = disease_name
    known_labels = known_labels[
        ["drug_id", "drug_name", "disease_id", "disease_name", "existing_relation"]
    ].sort_values("drug_name")

    candidates = (
        mechanism_profile.groupby(["drug_id", "drug_name"], as_index=False)
        .agg(
            matched_target_count=("protein_id", "nunique"),
            matched_targets=("protein_id", joined),
            matched_target_labels=("protein_label", joined),
            mechanism_actions=("action_type", joined),
            mechanism_sources=("source", joined),
            mechanism_evidence_ids=("evidence_id", joined),
        )
    )
    candidates["disease_id"] = disease_id
    candidates["disease_name"] = disease_name
    candidates = candidates.merge(
        known_labels[["drug_id", "existing_relation"]], on="drug_id", how="left"
    ).fillna("")
    candidates["candidate_status"] = "UNRESOLVED_RESEARCH_HYPOTHESIS"
    candidates.loc[
        candidates["existing_relation"].str.contains("contraindication", case=False, na=False),
        "candidate_status",
    ] = "CONTRAINDICATION_CONTROL_REVIEW"
    candidates.loc[
        candidates["existing_relation"].str.contains("indication|off-label use", case=False, na=False),
        "candidate_status",
    ] = "KNOWN_USE_CONTROL_NOT_NOVEL"
    candidates["initial_use_warning"] = (
        "This candidate pool is an evidence-retrieval set, not a treatment recommendation."
    )
    candidates = candidates.sort_values(
        ["candidate_status", "matched_target_count", "drug_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    candidates.insert(0, "candidate_id", [
        f"AD-V1-{index + 1:04d}" for index in range(len(candidates))
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entities_dir = args.output_dir / "entities"
    relations_dir = args.output_dir / "relations"
    evidence_dir = args.output_dir / "evidence_labels"
    provenance_dir = args.output_dir / "provenance"
    for directory in (entities_dir, relations_dir, evidence_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "disease_id": disease_id,
        "disease_name": disease_name,
        "disease_scope": "Alzheimer disease",
        "dataset_version": "alzheimers_v1",
    }]).to_csv(entities_dir / "disease.csv", index=False)
    target_profile.to_csv(entities_dir / "disease_target_profile.csv", index=False)
    mechanism_profile.to_csv(relations_dir / "drug_target_mechanisms.csv", index=False)
    known_labels.to_csv(evidence_dir / "known_alzheimers_relations.csv", index=False)
    candidates.to_csv(evidence_dir / "candidate_pool.csv", index=False)

    registry = pd.DataFrame([
        {
            "source_name": "ReMedy knowledge graph",
            "input_path": str(args.edges),
            "use": "Disease-protein graph context and existing drug-disease relations",
            "automatic_inference_allowed": "No therapeutic efficacy inference",
        },
        {
            "source_name": "ChEMBL mechanism records",
            "input_path": str(args.mechanisms),
            "use": "Drug target and action evidence",
            "automatic_inference_allowed": "No disease benefit inference",
        },
        {
            "source_name": "Open Targets genetic association context",
            "input_path": str(args.genetic_context),
            "use": "Genetic association provenance and direction context",
            "automatic_inference_allowed": "No therapeutic-direction inference",
        },
    ])
    registry.to_csv(provenance_dir / "source_registry.csv", index=False)

    manifest = {
        "dataset": "ReMedy Alzheimer disease curated V1",
        "disease_id": disease_id,
        "disease_name": disease_name,
        "disease_targets": int(len(target_profile)),
        "drug_target_mechanism_rows": int(len(mechanism_profile)),
        "candidate_pool_drugs": int(len(candidates)),
        "known_use_controls": int(candidates["candidate_status"].eq("KNOWN_USE_CONTROL_NOT_NOVEL").sum()),
        "contraindication_controls": int(candidates["candidate_status"].eq("CONTRAINDICATION_CONTROL_REVIEW").sum()),
        "unresolved_hypotheses": int(candidates["candidate_status"].eq("UNRESOLVED_RESEARCH_HYPOTHESIS").sum()),
        "policy": (
            "The dataset preserves source provenance and separates association, mechanism, "
            "known-use, and candidate records. No row is labelled therapeutically effective "
            "without source-cited outcome adjudication."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote Alzheimer disease curated V1 dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
