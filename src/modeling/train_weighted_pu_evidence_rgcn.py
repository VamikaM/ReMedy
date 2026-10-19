"""
Hybrid R-GCN training for ReMedy.

This model combines:
1. relation-aware graph embeddings from biological support edges, and
2. explicit shared-protein/shared-pathway features.

It keeps treatment and contraindication labels out of message passing.
Validation chooses the saved checkpoint; test is reported once per seed.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "model"
SPLIT_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "reports" / "model_evaluation"
CHECKPOINT_DIR = ROOT / "models"
PU_LABEL_PATH = MODEL_DIR / "v3_pu_train_labels.csv"

FEATURE_COLUMNS = [
    "shared_protein_count",
    "shared_protein_jaccard",
    "shared_pathway_count",
    "shared_pathway_jaccard",
    "drug_target_count",
    "disease_protein_count",
    "drug_pathway_count",
    "disease_pathway_count",
    "target_evidence_score",
    "specificity_weighted_target_support",
    "direction_compatible_target_count",
]
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,13,29,42,73")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--pu-ranking-weight", type=float, default=0.25,
                        help="Low-weight ranking signal: positives should rank above unlabelled pairs.")
    parser.add_argument("--hard-negative-weight", type=float, default=0.25,
                        help="Weight assigned to source-backed contraindication supervision.")
    parser.add_argument("--unlabelled-bce-weight", type=float, default=0.20,
                        help="Weak BCE weight for sampled unlabelled pairs; they are not assumed reliable negatives.")
    parser.add_argument("--max-hard-negatives-per-disease", type=int, default=20,
                        help="Caps contraindication pairs per disease per epoch to prevent disease-level label bias.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    k = min(10, len(labels))
    top_k = labels[order[:k]]
    positives = int(labels.sum())
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "auprc_average_precision": float(
            average_precision_score(labels, scores)
        ),
        "precision_at_10": float(top_k.mean()),
        "recall_at_10": float(top_k.sum() / positives) if positives else 0.0,
    }


class HybridRGCN(nn.Module):
    def __init__(self, num_nodes, num_types, hidden_dim, num_relations,
                 feature_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.entity_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.type_embedding = nn.Embedding(num_types, hidden_dim)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations,
                              num_bases=3)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations,
                              num_bases=3)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        pair = torch.cat(
            [drug, disease, drug * disease, torch.abs(drug - disease),
             explicit],
            dim=1,
        )
        return self.decoder(pair).squeeze(-1)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    graph = pd.read_parquet(MODEL_DIR / "training_graph_edges.parquet")
    mechanisms = pd.read_csv(DATA_DIR / "drug_mechanisms.csv", dtype={"drug_id": str, "protein_id": str})
    effects = pd.read_csv(DATA_DIR / "disease_target_effects.csv", dtype={"disease_id": str, "protein_id": str})
    nodes["node_id"] = nodes["node_id"].astype(str)
    graph["source_id"] = graph["source_id"].astype(str)
    graph["target_id"] = graph["target_id"].astype(str)
    mechanisms["drug_id"] = mechanisms["drug_id"].astype(str)
    mechanisms["protein_id"] = mechanisms["protein_id"].astype(str)
    effects["disease_id"] = effects["disease_id"].astype(str)
    effects["protein_id"] = effects["protein_id"].astype(str)
    if {"action_type"} - set(mechanisms.columns):
        raise ValueError("drug_mechanisms.csv must contain action_type")
    if {"disease_effect", "evidence_score"} - set(effects.columns):
        raise ValueError("disease_target_effects.csv must contain disease_effect and evidence_score")

    node_to_idx = {node_id: index for index, node_id in
                   enumerate(nodes["node_id"])}
    node_type_names = ["drug", "disease", "protein", "pathway"]
    node_type_to_idx = {name: index for index, name in
                        enumerate(node_type_names)}
    if set(nodes["node_type"]) - set(node_type_names):
        raise ValueError("Unexpected node types in nodes.parquet")

    node_type_tensor = torch.tensor(
        [node_type_to_idx[value] for value in nodes["node_type"]],
        dtype=torch.long, device=device
    )
    type_lookup = nodes.set_index("node_id")["node_type"].to_dict()
    drug_ids = set(nodes.loc[nodes["node_type"] == "drug", "node_id"])
    disease_ids = set(nodes.loc[nodes["node_type"] == "disease", "node_id"])

    relation_to_idx = {relation: index for index, relation in
                       enumerate(SUPPORT_RELATIONS)}
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
            relation_to_idx[row.relation] + len(SUPPORT_RELATIONS),
        ])

    edge_index = torch.tensor([source_indices, target_indices],
                              dtype=torch.long, device=device)
    edge_type = torch.tensor(relation_indices, dtype=torch.long,
                             device=device)

    drug_pathways = {
        drug: set().union(*(protein_pathways[p] for p in drug_targets[drug]))
        if drug_targets[drug] else set()
        for drug in drug_ids
    }
    disease_pathways = {
        disease: set().union(*(protein_pathways[p] for p in
                               disease_proteins[disease]))
        if disease_proteins[disease] else set()
        for disease in disease_ids
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

    def jaccard(left, right):
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def pair_features(frame):
        rows = []
        for row in frame.itertuples(index=False):
            drug_proteins = drug_targets[row.drug_id]
            disease_protein_set = disease_proteins[row.disease_id]
            drug_pathway_set = drug_pathways[row.drug_id]
            disease_pathway_set = disease_pathways[row.disease_id]
            shared_proteins = drug_proteins & disease_protein_set
            compatible_evidence_scores = []
            specificity_weighted_support = 0.0
            for protein in shared_proteins:
                effect = effect_rows.get((row.disease_id, protein))
                action_values = mechanism_signs.get((row.drug_id, protein), set())
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
            rows.append([
                len(shared_proteins),
                jaccard(drug_proteins, disease_protein_set),
                len(drug_pathway_set & disease_pathway_set),
                jaccard(drug_pathway_set, disease_pathway_set),
                len(drug_proteins),
                len(disease_protein_set),
                len(drug_pathway_set),
                len(disease_pathway_set),
                float(sum(compatible_evidence_scores)),
                float(specificity_weighted_support),
                len(compatible_evidence_scores),
            ])
        return np.asarray(rows, dtype=np.float32)

    pu_labels = pd.read_csv(
        PU_LABEL_PATH, dtype={"drug_id": str, "disease_id": str}
    )
    required_pu_columns = {"drug_id", "disease_id", "label_state", "training_target"}
    missing_pu_columns = required_pu_columns - set(pu_labels.columns)
    if missing_pu_columns:
        raise ValueError(
            f"v3_pu_train_labels.csv is missing columns: {sorted(missing_pu_columns)}"
        )
    for column in ("drug_id", "disease_id"):
        pu_labels[column] = pu_labels[column].astype(str)
    if pu_labels.duplicated(["drug_id", "disease_id"]).any():
        raise ValueError("v3_pu_train_labels.csv contains duplicate pairs")
    if any(type_lookup.get(x) != "drug" for x in pu_labels["drug_id"]):
        raise ValueError("Invalid drug IDs in v3_pu_train_labels.csv")
    if any(type_lookup.get(x) != "disease" for x in pu_labels["disease_id"]):
        raise ValueError("Invalid disease IDs in v3_pu_train_labels.csv")

    raw_splits = {}
    for name in ["validation", "test"]:
        frame = pd.read_csv(
            SPLIT_DIR / f"{name}_pairs.csv",
            dtype={"drug_id": str, "disease_id": str},
        )
        if frame.duplicated(["drug_id", "disease_id"]).any():
            raise ValueError(f"Duplicate pairs in {name}")
        if any(type_lookup.get(x) != "drug" for x in frame["drug_id"]):
            raise ValueError(f"Invalid drug IDs in {name}")
        if any(type_lookup.get(x) != "disease" for x in frame["disease_id"]):
            raise ValueError(f"Invalid disease IDs in {name}")
        raw_splits[name] = frame

    scaler = StandardScaler()
    pu_features = scaler.fit_transform(pair_features(pu_labels))
    split_features = {
        name: scaler.transform(pair_features(frame))
        for name, frame in raw_splits.items()
    }

    def make_pair_tensors(frame, feature_values, labels=None):
        tensors = (
            torch.tensor([node_to_idx[x] for x in frame["drug_id"]],
                         dtype=torch.long, device=device),
            torch.tensor([node_to_idx[x] for x in frame["disease_id"]],
                         dtype=torch.long, device=device),
            torch.tensor(feature_values, dtype=torch.float32, device=device),
        )
        if labels is None:
            return tensors
        return (*tensors, torch.tensor(labels, dtype=torch.float32, device=device))

    pu_tensors = make_pair_tensors(pu_labels, pu_features)
    tensors = {
        name: make_pair_tensors(
            frame, split_features[name],
            frame["label"].to_numpy(dtype=np.float32),
        )
        for name, frame in raw_splits.items()
    }
    positive_indices = np.flatnonzero(
        pu_labels["label_state"].eq("CONFIRMED_POSITIVE").to_numpy()
    )
    negative_indices = np.flatnonzero(
        pu_labels["label_state"].eq("RELIABLE_NEGATIVE").to_numpy()
    )
    unlabelled_indices = np.flatnonzero(
        pu_labels["label_state"].eq("UNLABELLED").to_numpy()
    )
    if min(len(positive_indices), len(negative_indices), len(unlabelled_indices)) == 0:
        raise ValueError(
            "Weighted PU training requires confirmed positives, reliable negatives, and unlabelled pairs."
        )
    negative_indices_by_disease = {
        disease_id: np.flatnonzero(
            pu_labels["label_state"].eq("RELIABLE_NEGATIVE").to_numpy()
            & pu_labels["disease_id"].eq(disease_id).to_numpy()
        )
        for disease_id in sorted(
            pu_labels.loc[
                pu_labels["label_state"].eq("RELIABLE_NEGATIVE"), "disease_id"
            ].unique()
        )
    }
    results = []
    champion = {"validation_ap": -1.0}

    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        set_seed(seed)
        model = HybridRGCN(
            len(nodes), len(node_type_names), args.hidden_dim,
            len(SUPPORT_RELATIONS) * 2, len(FEATURE_COLUMNS), args.dropout
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        best_state, best_epoch, best_ap, wait = None, 0, -1.0, 0

        for epoch in range(1, args.max_epochs + 1):
            # Contraindications are high-value safety evidence, but are not an
            # unbiased sample of ineffective treatments. Cap their contribution
            # per disease so frequent labels (for example epilepsy) cannot
            # become a disease-level shortcut for the model.
            negative_sample = np.concatenate([
                np.random.choice(
                    candidates,
                    size=min(len(candidates), args.max_hard_negatives_per_disease),
                    replace=False,
                )
                for candidates in negative_indices_by_disease.values()
            ])
            sample_size = len(positive_indices)
            unlabelled_sample = np.random.choice(
                unlabelled_indices, size=sample_size,
                replace=len(unlabelled_indices) < sample_size,
            )
            positive_tensor = torch.as_tensor(
                positive_indices, dtype=torch.long, device=device
            )
            negative_tensor = torch.as_tensor(
                negative_sample, dtype=torch.long, device=device
            )
            unlabelled_tensor = torch.as_tensor(
                unlabelled_sample, dtype=torch.long, device=device
            )

            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(node_type_tensor, edge_index, edge_type)
            positive_logits = model.decode(
                embeddings,
                pu_tensors[0][positive_tensor],
                pu_tensors[1][positive_tensor],
                pu_tensors[2][positive_tensor],
            )
            negative_logits = model.decode(
                embeddings,
                pu_tensors[0][negative_tensor],
                pu_tensors[1][negative_tensor],
                pu_tensors[2][negative_tensor],
            )
            unlabelled_logits = model.decode(
                embeddings,
                pu_tensors[0][unlabelled_tensor],
                pu_tensors[1][unlabelled_tensor],
                pu_tensors[2][unlabelled_tensor],
            )
            positive_loss = F.binary_cross_entropy_with_logits(
                positive_logits, torch.ones_like(positive_logits)
            )
            negative_loss = F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits)
            )
            # Unlabelled pairs may contain undiscovered positives. They receive
            # only weak negative pressure, unlike source-backed contraindications.
            unlabelled_bce_loss = F.binary_cross_entropy_with_logits(
                unlabelled_logits, torch.zeros_like(unlabelled_logits)
            )
            pu_ranking_loss = F.softplus(
                unlabelled_logits - positive_logits
            ).mean()
            loss = (
                positive_loss
                + args.hard_negative_weight * negative_loss
                + args.unlabelled_bce_weight * unlabelled_bce_loss
                + args.pu_ranking_weight * pu_ranking_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                embeddings = model.encode(
                    node_type_tensor, edge_index, edge_type
                )
                val_logits = model.decode(
                    embeddings, tensors["validation"][0],
                    tensors["validation"][1], tensors["validation"][2],
                )
                val_scores = torch.sigmoid(val_logits).cpu().numpy()
            validation_ap = average_precision_score(
                tensors["validation"][3].cpu().numpy(), val_scores
            )
            if validation_ap > best_ap + 1e-5:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch, best_ap, wait = epoch, float(validation_ap), 0
            else:
                wait += 1
            if wait >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(node_type_tensor, edge_index, edge_type)
            val_scores = torch.sigmoid(model.decode(
                embeddings, tensors["validation"][0],
                tensors["validation"][1], tensors["validation"][2],
            )).cpu().numpy()
            test_scores = torch.sigmoid(model.decode(
                embeddings, tensors["test"][0], tensors["test"][1],
                tensors["test"][2],
            )).cpu().numpy()

        row = {
            "seed": seed, "best_epoch": best_epoch,
            "best_validation_auprc": best_ap,
            **{f"validation_{key}": value for key, value in metrics(
                tensors["validation"][3].cpu().numpy(), val_scores
            ).items()},
            **{f"test_{key}": value for key, value in metrics(
                tensors["test"][3].cpu().numpy(), test_scores
            ).items()},
        }
        results.append(row)
        print(
            f"Seed {seed}: epoch {best_epoch}, validation AUPRC "
            f"{best_ap:.4f}, test AUPRC {row['test_auprc_average_precision']:.4f}"
        )

        if best_ap > champion["validation_ap"]:
            champion = {"validation_ap": best_ap, "seed": seed,
                        "epoch": best_epoch, "state": best_state}

    per_seed = pd.DataFrame(results)
    per_seed.to_csv(OUTPUT_DIR / "weighted_pu_evidence_rgcn_per_seed_metrics.csv",
                    index=False)
    summary = pd.DataFrame([{
        "model": "Weighted PU Evidence-Aware R-GCN",
        "runs": len(per_seed),
        "test_roc_auc_mean": per_seed["test_roc_auc"].mean(),
        "test_roc_auc_std": per_seed["test_roc_auc"].std(ddof=0),
        "test_auprc_mean": per_seed["test_auprc_average_precision"].mean(),
        "test_auprc_std": per_seed["test_auprc_average_precision"].std(ddof=0),
        "test_precision_at_10_mean": per_seed["test_precision_at_10"].mean(),
        "test_recall_at_10_mean": per_seed["test_recall_at_10"].mean(),
        "champion_seed": champion["seed"],
        "champion_epoch": champion["epoch"],
        "champion_validation_auprc": champion["validation_ap"],
    }])
    summary.to_csv(OUTPUT_DIR / "weighted_pu_evidence_rgcn_summary.csv", index=False)

    torch.save({
        "model_state_dict": champion["state"],
        "node_to_idx": node_to_idx,
        "node_type_to_idx": node_type_to_idx,
        "relation_to_idx": relation_to_idx,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "feature_columns": FEATURE_COLUMNS,
        "feature_scaler_mean": scaler.mean_.tolist(),
        "feature_scaler_scale": scaler.scale_.tolist(),
        "best_seed": champion["seed"],
        "best_epoch": champion["epoch"],
        "best_validation_auprc": champion["validation_ap"],
    }, CHECKPOINT_DIR / "weighted_pu_evidence_rgcn.pt")

    metadata = {
        "device": str(device),
        "support_relations": SUPPORT_RELATIONS,
        "treatment_labels_used_in_message_passing": False,
        "explicit_features": FEATURE_COLUMNS,
        "seeds": [int(x) for x in args.seeds.split(",") if x.strip()],
        "model_selection": "highest validation AUPRC",
        "training_method": "disease-balanced, weighted positive-unlabelled learning with capped contraindication supervision",
        "pu_label_source": str(PU_LABEL_PATH.relative_to(ROOT)),
        "pu_label_state_counts": pu_labels["label_state"].value_counts().to_dict(),
        "hard_negative_weight": args.hard_negative_weight,
        "unlabelled_bce_weight": args.unlabelled_bce_weight,
        "pu_ranking_weight": args.pu_ranking_weight,
        "max_hard_negatives_per_disease": args.max_hard_negatives_per_disease,
        "hard_negative_counts_by_disease": {
            disease: int(len(indices))
            for disease, indices in negative_indices_by_disease.items()
        },
    }
    (OUTPUT_DIR / "weighted_pu_evidence_rgcn_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nWeighted PU Evidence-Aware R-GCN completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
