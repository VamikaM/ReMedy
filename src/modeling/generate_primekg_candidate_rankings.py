"""Generate unseen drug-disease research hypotheses from the trained PrimeKG R-GCN."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "modeling"))
from train_primekg_rgcn import PrimeKGRGCN, build_graph, choose_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "model_v2" / "primekg")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "model_v2" / "primekg_rgcn.pt")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "model_v2")
    parser.add_argument("--top-k-per-disease", type=int, default=20)
    parser.add_argument("--pair-batch-size", type=int, default=100_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    nodes = pd.read_parquet(args.data_dir / "nodes.parquet").reset_index(drop=True)
    edges = pd.read_parquet(args.data_dir / "message_passing_edges.parquet")
    labels = pd.read_csv(args.data_dir / "drug_disease_labels_drug_cold.csv")

    node_type, edge_index, edge_type, type_vocab, relation_vocab = build_graph(nodes, edges, device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = PrimeKGRGCN(
        node_count=checkpoint["node_count"],
        node_type_count=checkpoint["node_type_count"],
        relation_count=checkpoint["relation_count"],
        hidden_dim=checkpoint["hidden_dim"],
        type_dim=checkpoint["type_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    drugs = nodes.loc[nodes["node_type"].eq("drug"), ["node_id", "node_name"]].copy()
    drugs["drug_index"] = drugs.index
    diseases = nodes.loc[nodes["node_type"].eq("disease"), ["node_id", "node_name"]].copy()
    diseases["disease_index"] = diseases.index
    known = set(zip(labels["drug_id"].astype(str), labels["disease_id"].astype(str)))

    with torch.no_grad():
        embeddings = model.encode(node_type, edge_index, edge_type)

    rows: list[pd.DataFrame] = []
    drug_indices = torch.tensor(drugs["drug_index"].to_numpy(), dtype=torch.long, device=device)
    for disease in diseases.itertuples(index=False):
        unknown_mask = ~drugs["node_id"].astype(str).map(lambda drug_id: (drug_id, str(disease.node_id)) in known)
        candidate_drugs = drugs.loc[unknown_mask].copy()
        candidate_indices = torch.tensor(candidate_drugs["drug_index"].to_numpy(), dtype=torch.long, device=device)
        disease_indices = torch.full_like(candidate_indices, int(disease.disease_index))
        score_chunks: list[torch.Tensor] = []
        for start in range(0, candidate_indices.numel(), args.pair_batch_size):
            stop = min(start + args.pair_batch_size, candidate_indices.numel())
            logits = model.decode(embeddings, candidate_indices[start:stop], disease_indices[start:stop])
            score_chunks.append(torch.sigmoid(logits).detach().cpu())
        candidate_drugs["model_score"] = torch.cat(score_chunks).numpy()
        top = candidate_drugs.nlargest(args.top_k_per_disease, "model_score").copy()
        top["disease_id"] = disease.node_id
        top["disease_name"] = disease.node_name
        top["candidate_status"] = "UNSEEN_PRIMEKG_TREATMENT_HYPOTHESIS"
        top["evidence_next_step"] = "Run ReMedy mechanism, safety, literature, and clinical-evidence review."
        rows.append(top)
        if len(rows) % 100 == 0:
            print(f"Scored {len(rows)}/{len(diseases)} diseases.", flush=True)

    ranking = pd.concat(rows, ignore_index=True).rename(columns={"node_id": "drug_id", "node_name": "drug_name"})
    ranking = ranking[[
        "disease_id", "disease_name", "drug_id", "drug_name", "model_score",
        "candidate_status", "evidence_next_step",
    ]].sort_values(["disease_id", "model_score"], ascending=[True, False])
    ranking["disease_rank"] = ranking.groupby("disease_id")["model_score"].rank(
        method="first", ascending=False
    ).astype(int)
    output = args.report_dir / "primekg_rgcn_unseen_top_candidates.csv"
    ranking.to_csv(output, index=False)
    metadata = {
        "model_path": str(args.model_path),
        "candidate_drugs": int(len(drugs)),
        "candidate_diseases": int(len(diseases)),
        "known_explicit_pairs_excluded": int(len(known)),
        "top_k_per_disease": args.top_k_per_disease,
        "rows_written": int(len(ranking)),
        "interpretation": "Scores rank unobserved PrimeKG treatment pairs for research review. The output is passed to ReMedy evidence evaluation before any shortlist status.",
    }
    (args.report_dir / "primekg_rgcn_candidate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote candidate hypotheses to {output}")


if __name__ == "__main__":
    main()
