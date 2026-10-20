"""Train a leakage-controlled relational GCN on the prepared PrimeKG ReMedy graph.

The input graph contains biological/contextual edges only. All indication, off-label,
and contraindication drug-disease relations are excluded from message passing and are
used solely as supervision labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.nn import RGCNConv


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "model_v2" / "primekg")
    parser.add_argument("--model-dir", type=Path, default=root / "models" / "model_v2")
    parser.add_argument("--report-dir", type=Path, default=root / "reports" / "model_v2")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--type-dim", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], dict[str, int]]:
    node_types = sorted(nodes["node_type"].fillna("unknown").astype(str).unique())
    type_to_index = {name: index for index, name in enumerate(node_types)}
    node_type = torch.tensor(
        nodes["node_type"].fillna("unknown").map(type_to_index).to_numpy(),
        dtype=torch.long,
        device=device,
    )

    relation_names = sorted(edges["relation"].fillna("unknown_relation").astype(str).unique())
    relation_to_index = {name: index for index, name in enumerate(relation_names)}
    forward_type = edges["relation"].fillna("unknown_relation").map(relation_to_index).to_numpy(dtype=np.int64)

    # Biological relations are directed in PrimeKG. Add explicit inverse relations so
    # the encoder can learn from target-to-source context without adding treatment edges.
    source = edges["source_index"].to_numpy(dtype=np.int64)
    target = edges["target_index"].to_numpy(dtype=np.int64)
    edge_index = torch.tensor(
        np.vstack([np.concatenate([source, target]), np.concatenate([target, source])]),
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor(
        np.concatenate([forward_type, forward_type + len(relation_names)]),
        dtype=torch.long,
        device=device,
    )
    relation_vocab = {name: index for index, name in enumerate(relation_names)}
    relation_vocab.update({f"inverse::{name}": index + len(relation_names) for index, name in enumerate(relation_names)})
    return node_type, edge_index, edge_type, type_to_index, relation_vocab


class PrimeKGRGCN(nn.Module):
    def __init__(self, node_count: int, node_type_count: int, relation_count: int, hidden_dim: int, type_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(node_count, hidden_dim)
        self.type_embedding = nn.Embedding(node_type_count, type_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim + type_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        bases = min(20, relation_count)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, relation_count, num_bases=bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, relation_count, num_bases=bases)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def encode(self, node_type: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        node_ids = torch.arange(node_type.numel(), device=node_type.device)
        x = self.input_projection(torch.cat([self.node_embedding(node_ids), self.type_embedding(node_type)], dim=1))
        x = torch.relu(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index, edge_type))
        return x

    def decode(self, embeddings: torch.Tensor, drug_index: torch.Tensor, disease_index: torch.Tensor) -> torch.Tensor:
        drug = embeddings[drug_index]
        disease = embeddings[disease_index]
        features = torch.cat([drug, disease, drug * disease, torch.abs(drug - disease)], dim=1)
        return self.decoder(features).squeeze(1)


def pair_tensors(frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(frame["drug_index"].to_numpy(), dtype=torch.long, device=device),
        torch.tensor(frame["disease_index"].to_numpy(), dtype=torch.long, device=device),
        torch.tensor(frame["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device),
    )


def metric_dict(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    y_true = labels.detach().cpu().numpy()
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    order = np.argsort(-scores)
    k = min(50, len(y_true))
    top = y_true[order[:k]]
    positives = max(int(y_true.sum()), 1)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "auprc_average_precision": float(average_precision_score(y_true, scores)),
        "precision_at_50": float(top.mean()),
        "recall_at_50": float(top.sum() / positives),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(args.data_dir / "nodes.parquet").reset_index(drop=True)
    edges = pd.read_parquet(args.data_dir / "message_passing_edges.parquet")
    train = pd.read_csv(args.data_dir / "train_pairs.csv")
    validation = pd.read_csv(args.data_dir / "validation_pairs.csv")
    test = pd.read_csv(args.data_dir / "test_pairs.csv")
    if nodes.index.to_numpy().tolist() != list(range(len(nodes))):
        raise RuntimeError("Node table index must be sequential.")
    if edges[["source_index", "target_index"]].isna().any().any():
        raise RuntimeError("Message-passing edge has an unresolved node index.")
    for name, frame in {"train": train, "validation": validation, "test": test}.items():
        if frame[["drug_index", "disease_index"]].isna().any().any():
            raise RuntimeError(f"{name} labels have unresolved node indices.")

    node_type, edge_index, edge_type, type_vocab, relation_vocab = build_graph(nodes, edges, device)
    tensors = {name: pair_tensors(frame, device) for name, frame in {"train": train, "validation": validation, "test": test}.items()}
    model = PrimeKGRGCN(
        node_count=len(nodes),
        node_type_count=len(type_vocab),
        relation_count=len(relation_vocab),
        hidden_dim=args.hidden_dim,
        type_dim=args.type_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    checkpoint_path = args.model_dir / "primekg_rgcn.pt"
    best_validation = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    print(json.dumps({
        "device": str(device),
        "nodes": len(nodes),
        "directed_message_passing_edges": int(edge_index.size(1)),
        "relation_types_with_inverse": len(relation_vocab),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "test_pairs": len(test),
    }, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.encode(node_type, edge_index, edge_type)
        train_logits = model.decode(embeddings, *tensors["train"][:2])
        train_loss = loss_fn(train_logits, tensors["train"][2])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(node_type, edge_index, edge_type)
            validation_logits = model.decode(embeddings, *tensors["validation"][:2])
            validation_metrics = metric_dict(tensors["validation"][2], validation_logits)
            train_metrics = metric_dict(tensors["train"][2], train_logits)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach().cpu()),
            "train_auprc": train_metrics["auprc_average_precision"],
            "validation_auprc": validation_metrics["auprc_average_precision"],
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_precision_at_50": validation_metrics["precision_at_50"],
            "validation_recall_at_50": validation_metrics["recall_at_50"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}: loss={record['train_loss']:.4f}, "
            f"validation AUPRC={record['validation_auprc']:.4f}, "
            f"ROC-AUC={record['validation_roc_auc']:.4f}",
            flush=True,
        )

        if validation_metrics["auprc_average_precision"] > best_validation + 1e-6:
            best_validation = validation_metrics["auprc_average_precision"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "node_count": len(nodes),
                "node_type_count": len(type_vocab),
                "relation_count": len(relation_vocab),
                "hidden_dim": args.hidden_dim,
                "type_dim": args.type_dim,
                "dropout": args.dropout,
                "type_vocab": type_vocab,
                "relation_vocab": relation_vocab,
                "best_epoch": best_epoch,
                "best_validation_auprc": best_validation,
                "training_policy": "Drug-disease indication/off-label/contraindication relations excluded from message passing.",
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(node_type, edge_index, edge_type)
        validation_logits = model.decode(embeddings, *tensors["validation"][:2])
        test_logits = model.decode(embeddings, *tensors["test"][:2])
    summary = {
        "model": "PrimeKG_RGCN",
        "best_epoch": best_epoch,
        "best_validation_auprc": best_validation,
        "validation": metric_dict(tensors["validation"][2], validation_logits),
        "test": metric_dict(tensors["test"][2], test_logits),
        "evaluation_design": "Held-out drugs have no supervised treatment labels in training; all treatment relations are excluded from graph message passing.",
        "label_definition": "Positive=PrimeKG explicit indication/off-label; negative=PrimeKG explicit contraindication.",
    }
    pd.DataFrame(history).to_csv(args.report_dir / "primekg_rgcn_training_history.csv", index=False)
    test_output = test[["drug_id", "drug_name", "disease_id", "disease_name", "label", "label_source"]].copy()
    test_output["score"] = torch.sigmoid(test_logits).detach().cpu().numpy()
    test_output.to_csv(args.report_dir / "primekg_rgcn_test_predictions.csv", index=False)
    (args.report_dir / "primekg_rgcn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
