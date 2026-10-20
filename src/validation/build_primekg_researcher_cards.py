"""Build evidence-aware researcher cards for PrimeKG R-GCN hypotheses.

This is a general, disease-agnostic gate. It does not promote a graph score by
itself: every ranked pair is checked for source-recorded drug-target and
disease-target support in the PrimeKG message-passing graph.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTEIN_TYPE = "gene/protein"
# PrimeKG drug-protein provenance distinguishes therapeutic targets from carrier and
# transporter context. Only the former can pass the automatic mechanism gate.
THERAPEUTIC_DRUG_RELATIONS = {"target", "enzyme"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "model_v2" / "primekg")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "model_v2")
    return parser.parse_args()


def add_relation(mapping: dict[str, set[str]], key: str, value: str) -> None:
    if key and value:
        mapping[key].add(value)


def make_endpoint_maps(edges: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], set[str]], dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    drug_targets: dict[str, set[str]] = defaultdict(set)
    disease_targets: dict[str, set[str]] = defaultdict(set)
    drug_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    disease_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    protein_neighbors: dict[str, set[str]] = defaultdict(set)

    for row in edges.itertuples(index=False):
        source_id, target_id = str(row.source_id), str(row.target_id)
        source_type, target_type = str(row.source_type), str(row.target_type)
        relation = str(row.relation)

        if source_type == "drug" and target_type == PROTEIN_TYPE:
            add_relation(drug_targets, source_id, target_id)
            add_relation(drug_relations, (source_id, target_id), relation)
        elif target_type == "drug" and source_type == PROTEIN_TYPE:
            add_relation(drug_targets, target_id, source_id)
            add_relation(drug_relations, (target_id, source_id), relation)

        if source_type == "disease" and target_type == PROTEIN_TYPE:
            add_relation(disease_targets, source_id, target_id)
            add_relation(disease_relations, (source_id, target_id), relation)
        elif target_type == "disease" and source_type == PROTEIN_TYPE:
            add_relation(disease_targets, target_id, source_id)
            add_relation(disease_relations, (target_id, source_id), relation)

        if source_type == PROTEIN_TYPE and target_type == PROTEIN_TYPE and relation == "protein protein":
            add_relation(protein_neighbors, source_id, target_id)
            add_relation(protein_neighbors, target_id, source_id)

    return drug_targets, disease_targets, drug_relations, disease_relations, protein_neighbors


def label_names(ids: set[str], protein_names: dict[str, str], limit: int = 8) -> str:
    return " | ".join(protein_names.get(identifier, identifier) for identifier in sorted(ids)[:limit])


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.report_dir / "primekg_rgcn_unseen_top_candidates.csv"
    ranking = pd.read_csv(ranking_path, dtype={"drug_id": str, "disease_id": str})
    nodes = pd.read_parquet(args.data_dir / "nodes.parquet")
    edges = pd.read_parquet(args.data_dir / "message_passing_edges.parquet")

    proteins = nodes.loc[nodes["node_type"].eq(PROTEIN_TYPE), ["node_id", "node_name"]].drop_duplicates("node_id")
    protein_names = dict(zip(proteins["node_id"].astype(str), proteins["node_name"].fillna("").astype(str)))
    (
        drug_targets,
        disease_targets,
        drug_relations,
        disease_relations,
        protein_neighbors,
    ) = make_endpoint_maps(edges)

    cards: list[dict[str, object]] = []
    for row in ranking.itertuples(index=False):
        drug_id, disease_id = str(row.drug_id), str(row.disease_id)
        all_targets = drug_targets.get(drug_id, set())
        disease_proteins = disease_targets.get(disease_id, set())
        therapeutic_targets = {
            protein for protein in all_targets
            if drug_relations.get((drug_id, protein), set()) & THERAPEUTIC_DRUG_RELATIONS
        }
        shared_all = all_targets & disease_proteins
        direct = therapeutic_targets & disease_proteins
        binding_context_only = shared_all - direct
        neighborhood: set[str] = set()
        for protein in therapeutic_targets:
            neighborhood.update(protein_neighbors.get(protein, set()))
        bridge = (neighborhood & disease_proteins) - direct

        if direct:
            status = "MECHANISM_REVIEW_READY"
            next_action = "Retrieve disease-specific literature, trial outcomes, exposure, and safety evidence for the direct therapeutic target path."
            priority = 3
        elif bridge:
            status = "NETWORK_CONTEXT_REVIEW_REQUIRED"
            next_action = "Verify whether the one-hop protein network bridge is disease-causal before external evidence retrieval."
            priority = 2
        elif binding_context_only:
            status = "WITHHELD_NONTHERAPEUTIC_BINDING_CONTEXT"
            next_action = "Withhold: overlap is limited to carrier/transporter binding context rather than a recorded therapeutic target action."
            priority = 1
        else:
            status = "WITHHELD_NO_DISEASE_TARGET_SUPPORT"
            next_action = "Withhold from researcher shortlist; no direct therapeutic target or one-hop disease-target support was retrieved."
            priority = 0

        direct_drug_actions = set()
        direct_disease_links = set()
        for protein in shared_all:
            direct_drug_actions.update(drug_relations.get((drug_id, protein), set()))
            direct_disease_links.update(disease_relations.get((disease_id, protein), set()))

        cards.append({
            "disease_id": disease_id,
            "disease_name": row.disease_name,
            "disease_rank": int(row.disease_rank),
            "drug_id": drug_id,
            "drug_name": row.drug_name,
            "model_score": float(row.model_score),
            "known_pair_control": "EXCLUDED_FROM_OUTPUT_IF_EXPLICIT_INDICATION_OFF_LABEL_OR_CONTRAINDICATION",
            "drug_target_count_all_relations": len(all_targets),
            "drug_therapeutic_target_count": len(therapeutic_targets),
            "disease_target_count": len(disease_proteins),
            "direct_shared_target_count": len(direct),
            "direct_shared_targets": label_names(direct, protein_names),
            "nontherapeutic_binding_overlap_count": len(binding_context_only),
            "nontherapeutic_binding_overlap_targets": label_names(binding_context_only, protein_names),
            "direct_drug_relation_types": " | ".join(sorted(direct_drug_actions)),
            "direct_disease_relation_types": " | ".join(sorted(direct_disease_links)),
            "one_hop_network_bridge_count": len(bridge),
            "one_hop_network_bridge_targets": label_names(bridge, protein_names),
            "mechanism_gate_status": status,
            "mechanism_gate_priority": priority,
            "researcher_next_action": next_action,
        })

    result = pd.DataFrame(cards)
    result = result.sort_values(
        ["disease_id", "mechanism_gate_priority", "model_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    result["research_priority_rank"] = (
        result.groupby("disease_id").cumcount() + 1
    )
    result = result.drop(columns="mechanism_gate_priority")
    output_path = args.report_dir / "primekg_rgcn_researcher_cards.csv"
    result.to_csv(output_path, index=False)

    summary = {
        "input_ranked_hypotheses": int(len(result)),
        "diseases": int(result["disease_id"].nunique()),
        "mechanism_gate_status_counts": result["mechanism_gate_status"].value_counts().to_dict(),
        "policy": {
            "known_treatment_or_contraindication": "excluded before ranking",
            "direct_therapeutic_target": "eligible for external evidence retrieval",
            "carrier_or_transporter_only_overlap": "withheld from researcher shortlist",
            "one_hop_network_context": "requires causal-mechanism review",
            "no_disease_target_support": "withheld from researcher shortlist",
        },
    }
    (args.report_dir / "primekg_rgcn_researcher_cards_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote universal researcher cards to {output_path}")


if __name__ == "__main__":
    main()
