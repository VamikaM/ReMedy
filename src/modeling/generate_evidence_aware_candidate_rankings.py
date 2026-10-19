"""
Generate novelty-ready candidate rankings from a trained ReMedy Evidence-Aware R-GCN.

Run after train_hybrid_rgcn.py.  The script scores only drug-disease pairs
without an indication, off-label-use, or contraindication edge in the frozen KG.
"""
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "model"
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"
CHECKPOINT_PATH = ROOT / "models" / "evidence_aware_rgcn.pt"
SUPPORT_RELATIONS = ["targets", "participates_in", "associated_with"]
NEGATIVE_ACTION_TERMS = ("INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE", "SUPPRESSOR")
POSITIVE_ACTION_TERMS = ("AGONIST", "ACTIVATOR", "POSITIVE", "POTENTIATOR", "INDUCER")
POSITIVE_EFFECTS = {"INCREASED_ACTIVITY", "GAIN_OF_FUNCTION", "HARMFUL_GAIN", "RISK_GAIN", "OVERACTIVE", "UPREGULATED"}
NEGATIVE_EFFECTS = {"DECREASED_ACTIVITY", "LOSS_OF_FUNCTION", "HARMFUL_LOSS", "PROTECTIVE_LOSS", "UNDERACTIVE", "DOWNREGULATED"}

def action_sign(value):
    action = str(value).upper().replace("-", "_").replace(" ", "_")
    if any(term in action for term in NEGATIVE_ACTION_TERMS):
        return -1
    if any(term in action for term in POSITIVE_ACTION_TERMS):
        return 1
    return 0

def effect_sign(value):
    effect = str(value).upper().replace("-", "_").replace(" ", "_")
    if effect in POSITIVE_EFFECTS:
        return 1
    if effect in NEGATIVE_EFFECTS:
        return -1
    return 0


class HybridRGCN(nn.Module):
    def __init__(self, num_nodes, num_types, hidden_dim, num_relations,
                 feature_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.entity_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations,
                              num_bases=3)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations,
                              num_bases=3)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def encode(self, node_type_ids, edge_index, edge_type):
        x = self.entity_embedding.weight + self.type_embedding(node_type_ids)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_type)

    def decode(self, embeddings, drug_indices, disease_indices, features):
        drug = embeddings[drug_indices]
        disease = embeddings[disease_indices]
        explicit = self.feature_encoder(features)
        joined = torch.cat(
            [drug, disease, drug * disease, torch.abs(drug - disease),
             explicit], dim=1
        )
        return self.decoder(joined).squeeze(-1)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Missing models/evidence_aware_rgcn.pt. Run train_evidence_aware_rgcn.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,
                            weights_only=False)

    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    graph = pd.read_parquet(MODEL_DIR / "training_graph_edges.parquet")
    mechanisms = pd.read_csv(DATA_DIR / "drug_mechanisms.csv", dtype={"drug_id": str, "protein_id": str})
    effects = pd.read_csv(DATA_DIR / "disease_target_effects.csv", dtype={"disease_id": str, "protein_id": str})
    for frame in [nodes, edges, graph, mechanisms, effects]:
        for column in ["node_id", "source_id", "target_id"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

    if {"action_type"} - set(mechanisms.columns):
        raise ValueError("drug_mechanisms.csv must contain action_type")
    if {"disease_effect", "evidence_score"} - set(effects.columns):
        raise ValueError("disease_target_effects.csv must contain disease_effect and evidence_score")

    node_to_idx = checkpoint["node_to_idx"]
    if set(node_to_idx) != set(nodes["node_id"]):
        raise ValueError("Checkpoint does not match the current node table.")

    type_by_id = nodes.set_index("node_id")["node_type"].to_dict()
    ordered_nodes = [node for node, _ in sorted(
        node_to_idx.items(), key=lambda item: item[1]
    )]
    node_type_tensor = torch.tensor(
        [checkpoint["node_type_to_idx"][type_by_id[node]]
         for node in ordered_nodes],
        dtype=torch.long, device=device
    )

    relation_to_idx = checkpoint["relation_to_idx"]
    source_indices, target_indices, relation_indices = [], [], []
    drug_targets = defaultdict(set)
    disease_proteins = defaultdict(set)
    protein_pathways = defaultdict(set)

    for row in graph.itertuples(index=False):
        if row.relation not in relation_to_idx:
            continue
        if row.relation == "targets":
            drug_targets[row.source_id].add(row.target_id)
        elif row.relation == "associated_with":
            disease_proteins[row.source_id].add(row.target_id)
        else:
            protein_pathways[row.source_id].add(row.target_id)

        source_indices.extend([node_to_idx[row.source_id],
                               node_to_idx[row.target_id]])
        target_indices.extend([node_to_idx[row.target_id],
                               node_to_idx[row.source_id]])
        relation_indices.extend([
            relation_to_idx[row.relation],
            relation_to_idx[row.relation] + len(relation_to_idx),
        ])

    edge_index = torch.tensor([source_indices, target_indices],
                              dtype=torch.long, device=device)
    edge_type = torch.tensor(relation_indices, dtype=torch.long,
                             device=device)

    drugs = sorted(nodes.loc[nodes["node_type"] == "drug", "node_id"])
    diseases = sorted(nodes.loc[nodes["node_type"] == "disease", "node_id"])
    drug_pathways = {
        drug: set().union(*(protein_pathways[p] for p in drug_targets[drug]))
        if drug_targets[drug] else set()
        for drug in drugs
    }
    disease_pathways = {
        disease: set().union(*(protein_pathways[p] for p in
                               disease_proteins[disease]))
        if disease_proteins[disease] else set()
        for disease in diseases
    }
    protein_drug_degree = defaultdict(int)
    protein_disease_degree = defaultdict(int)
    for values in drug_targets.values():
        for protein in values:
            protein_drug_degree[protein] += 1
    for values in disease_proteins.values():
        for protein in values:
            protein_disease_degree[protein] += 1
    mechanism_signs = defaultdict(set)
    for row in mechanisms.itertuples(index=False):
        sign = action_sign(row.action_type)
        if sign:
            mechanism_signs[(row.drug_id, row.protein_id)].add(sign)
    effects["evidence_score"] = pd.to_numeric(effects["evidence_score"], errors="coerce").fillna(0.0)
    effect_rows = {}
    for row in effects.itertuples(index=False):
        sign = effect_sign(row.disease_effect)
        if sign:
            key = (row.disease_id, row.protein_id)
            current = effect_rows.get(key)
            if current is None or row.evidence_score > current[1]:
                effect_rows[key] = (sign, float(row.evidence_score))

    known_pairs = set(edges.loc[
        edges["relation"].isin(
            ["indication", "off-label use", "contraindication"]
        ), ["source_id", "target_id"]
    ].itertuples(index=False, name=None))
    candidates = [(drug, disease) for drug in drugs for disease in diseases
                  if (drug, disease) not in known_pairs]

    def jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    features = []
    for drug, disease in candidates:
        targets = drug_targets[drug]
        disease_set = disease_proteins[disease]
        drug_set = drug_pathways[drug]
        pathway_set = disease_pathways[disease]
        shared_proteins = targets & disease_set
        compatible_evidence_scores = []
        specificity_weighted_support = 0.0
        for protein in shared_proteins:
            effect = effect_rows.get((disease, protein))
            action_values = mechanism_signs.get((drug, protein), set())
            if effect is None or not action_values:
                continue
            disease_sign, evidence_score = effect
            # A compatible action counteracts the disease-associated direction.
            if any(action * disease_sign == -1 for action in action_values):
                compatible_evidence_scores.append(evidence_score)
                specificity_weighted_support += evidence_score / np.sqrt(
                    max(1, protein_drug_degree[protein])
                    * max(1, protein_disease_degree[protein])
                )
        features.append([
            len(shared_proteins), jaccard(targets, disease_set),
            len(drug_set & pathway_set), jaccard(drug_set, pathway_set),
            len(targets), len(disease_set), len(drug_set), len(pathway_set),
            float(sum(compatible_evidence_scores)),
            float(specificity_weighted_support),
            len(compatible_evidence_scores),
        ])
    raw_features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(checkpoint["feature_scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scaler_scale"], dtype=np.float32)
    features = (raw_features - mean) / np.where(scale == 0, 1.0, scale)

    model = HybridRGCN(
        num_nodes=len(node_to_idx),
        num_types=len(checkpoint["node_type_to_idx"]),
        hidden_dim=checkpoint["hidden_dim"],
        num_relations=len(relation_to_idx) * 2,
        feature_dim=len(checkpoint["feature_columns"]),
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores = []
    batch_size = 1024
    with torch.no_grad():
        embeddings = model.encode(node_type_tensor, edge_index, edge_type)
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            drug_index = torch.tensor([node_to_idx[d] for d, _ in batch],
                                      dtype=torch.long, device=device)
            disease_index = torch.tensor([node_to_idx[x] for _, x in batch],
                                         dtype=torch.long, device=device)
            feature_batch = torch.tensor(
                features[start:start + batch_size], dtype=torch.float32,
                device=device
            )
            scores.extend(torch.sigmoid(model.decode(
                embeddings, drug_index, disease_index, feature_batch
            )).cpu().numpy().tolist())

    labels = nodes.set_index("node_id")["label"].to_dict()
    ranking = pd.DataFrame({
        "drug_id": [drug for drug, _ in candidates],
        "drug_name": [labels[drug] for drug, _ in candidates],
        "disease_id": [disease for _, disease in candidates],
        "disease_name": [labels[disease] for _, disease in candidates],
        "evidence_aware_rgcn_score": scores,
        "shared_protein_count": raw_features[:, 0],
        "shared_pathway_count": raw_features[:, 2],
        "target_evidence_score": raw_features[:, 8],
        "specificity_weighted_target_support": raw_features[:, 9],
        "direction_compatible_target_count": raw_features[:, 10],
    }).sort_values(
        ["hybrid_rgcn_score", "drug_id", "disease_id"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    # Raw scores are an internal model artifact.  They are intentionally not
    # emitted as a researcher-facing top-candidate list: run
    # src/validation/evidence_gate.py after enriching mechanism, disease-effect,
    # and safety-context inputs.
    raw_output = OUTPUT_DIR / "evidence_aware_rgcn_raw_internal_scores.csv"
    ranking.to_csv(raw_output, index=False)

    metadata = {
        "candidate_count": len(ranking),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "best_seed": checkpoint["best_seed"],
        "best_validation_auprc": checkpoint["best_validation_auprc"],
        "training_graph": str((MODEL_DIR / "training_graph_edges.parquet")
                              .relative_to(ROOT)),
        "score_interpretation": "Internal graph-model signal; not an eligibility or clinical-use score",
        "publication_status": "internal_audit_only",
        "next_step": "Run evidence_gate.py using this ranking; only mechanism-compatible pairs may enter evidence review.",
    }
    (OUTPUT_DIR / "evidence_aware_rgcn_candidate_ranking_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Candidate pairs scored internally:", len(ranking))
    print("Raw scores saved for audit only:", raw_output)
    print("Next: run src/validation/evidence_gate.py before displaying candidates.")


if __name__ == "__main__":
    main()
