"""Build a leakage-safe PrimeKG training graph for ReMedy drug repurposing.

The builder streams PrimeKG in chunks, uses only explicit indication/off-label
relations as positive labels and explicit contraindications as negatives, removes
all drug-disease label edges from message passing, and splits by held-out drug.
This prevents a model from seeing the answer edge during graph propagation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_TERMS = {"indication", "off label use", "off-label use", "offlabel use"}
NEGATIVE_TERMS = {"contraindication", "contraindicated"}


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def norm(value: object) -> str:
    return clean(value).lower().replace("_", " ").replace("-", " ")


def drug_disease_pair(row: pd.Series) -> dict | None:
    x_type, y_type = norm(row.x_type), norm(row.y_type)
    relation = norm(row.display_relation)
    if {x_type, y_type} != {"drug", "disease"}:
        return None
    if relation in POSITIVE_TERMS:
        label, source = 1, "PrimeKG_explicit_" + relation.replace(" ", "_")
    elif relation in NEGATIVE_TERMS:
        label, source = 0, "PrimeKG_explicit_contraindication"
    else:
        return None
    if x_type == "drug":
        drug_id, drug_name = clean(row.x_id), clean(row.x_name)
        disease_id, disease_name = clean(row.y_id), clean(row.y_name)
    else:
        drug_id, drug_name = clean(row.y_id), clean(row.y_name)
        disease_id, disease_name = clean(row.x_id), clean(row.x_name)
    if not drug_id or not disease_id:
        return None
    return {
        "drug_id": drug_id, "drug_name": drug_name,
        "disease_id": disease_id, "disease_name": disease_name,
        "label": label, "label_source": source,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primekg", type=Path, default=root / "data" / "external" / "primekg" / "kg.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "model_v2" / "primekg")
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.primekg.exists():
        raise FileNotFoundError(f"PrimeKG file not found: {args.primekg}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict] = []
    relation_counts: dict[tuple[str, str, str], int] = {}
    for chunk_index, chunk in enumerate(
        pd.read_csv(args.primekg, chunksize=args.chunk_size, low_memory=False), start=1
    ):
        for row in chunk.itertuples(index=False):
            key = (norm(row.x_type), norm(row.display_relation), norm(row.y_type))
            relation_counts[key] = relation_counts.get(key, 0) + 1
            pair = drug_disease_pair(row)
            if pair:
                pair_rows.append(pair)
        print(f"Scanned PrimeKG label chunk {chunk_index}; retained {len(pair_rows)} explicit pairs.", flush=True)

    labels = pd.DataFrame(pair_rows).drop_duplicates(["drug_id", "disease_id", "label"])
    # Resolve rare contradictory source relations conservatively: exclude them.
    conflict_keys = labels.groupby(["drug_id", "disease_id"])["label"].nunique()
    conflicts = set(conflict_keys[conflict_keys.gt(1)].index)
    if conflicts:
        labels = labels.loc[
            ~labels.set_index(["drug_id", "disease_id"]).index.isin(conflicts)
        ].copy()
    positives = labels.loc[labels["label"].eq(1)].copy()
    negatives = labels.loc[labels["label"].eq(0)].copy()
    if positives.empty or negatives.empty:
        raise RuntimeError("PrimeKG did not yield both explicit positive and contraindication labels.")

    # Drug-cold split: a held-out drug never contributes a treatment label in training.
    rng = np.random.default_rng(args.seed)
    drugs = np.array(sorted(positives["drug_id"].unique()))
    rng.shuffle(drugs)
    n_train = int(len(drugs) * 0.70)
    n_val = int(len(drugs) * 0.85)
    split_by_drug = {
        **{drug: "train" for drug in drugs[:n_train]},
        **{drug: "validation" for drug in drugs[n_train:n_val]},
        **{drug: "test" for drug in drugs[n_val:]},
    }
    labels = labels.loc[labels["drug_id"].isin(split_by_drug)].copy()
    labels["split"] = labels["drug_id"].map(split_by_drug)

    # Balance explicit contraindications against indications separately in each split.
    balanced: list[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        subset = labels.loc[labels["split"].eq(split)]
        pos = subset.loc[subset["label"].eq(1)]
        neg = subset.loc[subset["label"].eq(0)]
        neg = neg.sample(n=min(len(pos), len(neg)), random_state=args.seed) if len(neg) > len(pos) else neg
        balanced.append(pd.concat([pos, neg], ignore_index=True))
    labels = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_drugs = set(labels.loc[labels["split"].eq("train"), "drug_id"])
    all_label_drugs = set(labels["drug_id"])
    all_label_diseases = set(labels["disease_id"])

    # Second streaming pass: keep only biological/contextual edges incident to labelled
    # drugs or diseases. All drug-disease edges are excluded to remove label leakage.
    graph_rows: list[dict] = []
    for chunk_index, chunk in enumerate(
        pd.read_csv(args.primekg, chunksize=args.chunk_size, low_memory=False), start=1
    ):
        for row in chunk.itertuples(index=False):
            x_type, y_type = norm(row.x_type), norm(row.y_type)
            if {x_type, y_type} == {"drug", "disease"}:
                continue
            x_id, y_id = clean(row.x_id), clean(row.y_id)
            # For held-out drugs, do not retain any incident edge; this is a strict
            # drug-cold evaluation rather than an easier transductive test.
            if x_type == "drug" and x_id not in train_drugs:
                continue
            if y_type == "drug" and y_id not in train_drugs:
                continue
            incident_to_scope = (
                x_id in train_drugs or y_id in train_drugs
                or x_id in all_label_diseases or y_id in all_label_diseases
            )
            if not incident_to_scope or not x_id or not y_id:
                continue
            graph_rows.append({
                "source_id": x_id, "source_type": x_type, "source_name": clean(row.x_name),
                "relation": norm(row.display_relation) or norm(row.relation),
                "target_id": y_id, "target_type": y_type, "target_name": clean(row.y_name),
                "source_database": clean(row.x_source), "target_database": clean(row.y_source),
            })
        print(f"Scanned PrimeKG graph chunk {chunk_index}; retained {len(graph_rows)} biological edges.", flush=True)

    edges = pd.DataFrame(graph_rows).drop_duplicates(["source_id", "relation", "target_id"])
    if edges.empty:
        raise RuntimeError("No biological graph edges were retained; inspect PrimeKG relation schema.")
    nodes = pd.concat([
        edges[["source_id", "source_type", "source_name", "source_database"]].rename(
            columns={"source_id": "node_id", "source_type": "node_type", "source_name": "node_name", "source_database": "source_database"}
        ),
        edges[["target_id", "target_type", "target_name", "target_database"]].rename(
            columns={"target_id": "node_id", "target_type": "node_type", "target_name": "node_name", "target_database": "source_database"}
        ),
    ], ignore_index=True).drop_duplicates("node_id")

    # Ensure every pair endpoint exists even when it has no biological edge.
    pair_nodes = pd.concat([
        labels[["drug_id", "drug_name"]].rename(columns={"drug_id": "node_id", "drug_name": "node_name"}).assign(node_type="drug", source_database="PrimeKG"),
        labels[["disease_id", "disease_name"]].rename(columns={"disease_id": "node_id", "disease_name": "node_name"}).assign(node_type="disease", source_database="PrimeKG"),
    ], ignore_index=True).drop_duplicates("node_id")
    nodes = pd.concat([nodes, pair_nodes], ignore_index=True).drop_duplicates("node_id", keep="first")
    node_map = {node_id: index for index, node_id in enumerate(nodes["node_id"])}
    edges["source_index"] = edges["source_id"].map(node_map)
    edges["target_index"] = edges["target_id"].map(node_map)
    labels["drug_index"] = labels["drug_id"].map(node_map)
    labels["disease_index"] = labels["disease_id"].map(node_map)

    labels.to_csv(args.output_dir / "drug_disease_labels_drug_cold.csv", index=False)
    for split in ["train", "validation", "test"]:
        labels.loc[labels["split"].eq(split)].to_csv(
            args.output_dir / f"{split}_pairs.csv", index=False
        )
    edges.to_parquet(args.output_dir / "message_passing_edges.parquet", index=False)
    nodes.to_parquet(args.output_dir / "nodes.parquet", index=False)
    pd.DataFrame(
        [{"x_type": x, "display_relation": r, "y_type": y, "count": count}
         for (x, r, y), count in relation_counts.items()]
    ).sort_values("count", ascending=False).to_csv(args.output_dir / "relation_schema_inventory.csv", index=False)

    summary = {
        "primekg_source": str(args.primekg),
        "explicit_indication_or_off_label_positive_pairs": int((labels["label"] == 1).sum()),
        "explicit_contraindication_negative_pairs": int((labels["label"] == 0).sum()),
        "conflicting_pairs_excluded": len(conflicts),
        "unique_drugs_in_drug_cold_splits": int(labels["drug_id"].nunique()),
        "unique_diseases": int(labels["disease_id"].nunique()),
        "nodes": int(len(nodes)),
        "message_passing_edges_without_drug_disease_labels": int(len(edges)),
        "split_counts": labels.groupby("split").size().to_dict(),
        "policy": (
            "No drug-disease indication, off-label, or contraindication edge is included "
            "in message passing. Test-drug incident biological edges are also withheld."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
