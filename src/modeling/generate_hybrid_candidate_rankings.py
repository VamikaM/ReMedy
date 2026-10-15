"""
Generate evidence-ready candidate rankings from a trained ReMedy Hybrid R-GCN.

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
CHECKPOINT_PATH = ROOT / "models" / "hybrid_rgcn.pt"
SUPPORT_RELATIONS = ["targets", "participates_in", "associated_with"]


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
            "Missing models/hybrid_rgcn.pt. Run train_hybrid_rgcn.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,
                            weights_only=False)

    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    graph = pd.read_parquet(MODEL_DIR / "training_graph_edges.parquet")
    for frame in [nodes, edges, graph]:
        for column in ["node_id", "source_id", "target_id"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(str)

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
        features.append([
            len(targets & disease_set), jaccard(targets, disease_set),
            len(drug_set & pathway_set), jaccard(drug_set, pathway_set),
            len(targets), len(disease_set), len(drug_set), len(pathway_set),
        ])
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(checkpoint["feature_scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scaler_scale"], dtype=np.float32)
    features = (features - mean) / np.where(scale == 0, 1.0, scale)

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
        "hybrid_rgcn_score": scores,
        "shared_protein_count": features[:, 0] * scale[0] + mean[0],
        "shared_pathway_count": features[:, 2] * scale[2] + mean[2],
    }).sort_values(
        ["hybrid_rgcn_score", "drug_id", "disease_id"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    ranking.to_csv(OUTPUT_DIR / "hybrid_rgcn_unobserved_candidate_ranking.csv",
                   index=False)
    ranking.head(100).to_csv(
        OUTPUT_DIR / "hybrid_rgcn_top_100_candidates.csv", index=False
    )

    metadata = {
        "candidate_count": len(ranking),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "best_seed": checkpoint["best_seed"],
        "best_validation_auprc": checkpoint["best_validation_auprc"],
        "training_graph": str((MODEL_DIR / "training_graph_edges.parquet")
                              .relative_to(ROOT)),
        "score_interpretation": "Model ranking score for unobserved pairs",
    }
    (OUTPUT_DIR / "hybrid_rgcn_candidate_ranking_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Candidate pairs scored:", len(ranking))
    print(ranking.head(20)[
        ["rank", "drug_name", "disease_name", "hybrid_rgcn_score",
         "shared_protein_count", "shared_pathway_count"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
