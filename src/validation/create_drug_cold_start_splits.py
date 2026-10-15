"""
Create drug-cold-start link-prediction splits for ReMedy.

A drug appears in exactly one of train, validation, or test.  The model may
still see that drug's biological target/pathway context, but never sees one of
its treatment labels during training for validation/test drugs.
"""
from pathlib import Path
import json
import random

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits_drug_cold"
MODEL_DIR = ROOT / "data" / "model"

TREATMENT_RELATIONS = {"indication", "off-label use"}
CONTRAINDICATION = "contraindication"


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    evidence = pd.read_parquet(DATA_DIR / "evidence.parquet")

    node_types = nodes.set_index("node_id")["node_type"].to_dict()
    drugs = sorted(nodes.loc[nodes["node_type"] == "drug", "node_id"])
    diseases = sorted(nodes.loc[nodes["node_type"] == "disease", "node_id"])

    def pairs_for(relation):
        frame = edges.loc[edges["relation"] == relation,
                          ["source_id", "target_id"]].drop_duplicates()
        for row in frame.itertuples(index=False):
            if node_types[row.source_id] != "drug":
                raise ValueError("Drug-disease source type mismatch")
            if node_types[row.target_id] != "disease":
                raise ValueError("Drug-disease target type mismatch")
        return set(frame.itertuples(index=False, name=None))

    indications = pairs_for("indication")
    off_label = pairs_for("off-label use")
    contraindications = pairs_for(CONTRAINDICATION)
    treatment = indications | off_label
    conflicts = treatment & contraindications
    positives = treatment - contraindications

    positive_drugs = sorted({drug for drug, _ in positives})
    rng = random.Random(SEED)
    rng.shuffle(positive_drugs)

    n_drugs = len(positive_drugs)
    n_train_drugs = int(n_drugs * 0.70)
    n_validation_drugs = int(n_drugs * 0.15)
    drug_groups = {
        "train": set(positive_drugs[:n_train_drugs]),
        "validation": set(
            positive_drugs[n_train_drugs:n_train_drugs + n_validation_drugs]
        ),
        "test": set(positive_drugs[n_train_drugs + n_validation_drugs:]),
    }

    # A positive drug must belong to one and only one split.
    if sum(len(group) for group in drug_groups.values()) != n_drugs:
        raise AssertionError("Drug split allocation failed")
    if (drug_groups["train"] & drug_groups["validation"] or
            drug_groups["train"] & drug_groups["test"] or
            drug_groups["validation"] & drug_groups["test"]):
        raise AssertionError("Drug overlap detected")

    positive_splits = {
        name: sorted(pair for pair in positives if pair[0] in group)
        for name, group in drug_groups.items()
    }

    all_known = treatment | contraindications
    negative_pool_by_split = {}
    for name, group in drug_groups.items():
        candidates = [
            (drug, disease)
            for drug in sorted(group)
            for disease in diseases
            if (drug, disease) not in all_known
        ]
        rng.shuffle(candidates)
        if len(candidates) < len(positive_splits[name]):
            raise ValueError(f"Insufficient negative candidates for {name}")
        negative_pool_by_split[name] = candidates[:len(positive_splits[name])]

    def label_source(pair):
        labels = []
        if pair in indications:
            labels.append("indication")
        if pair in off_label:
            labels.append("off-label use")
        return ";".join(labels)

    summary = {}
    for name in ["train", "validation", "test"]:
        rows = [
            {
                "drug_id": drug, "disease_id": disease, "label": 1,
                "label_source": label_source((drug, disease)), "split": name,
            }
            for drug, disease in positive_splits[name]
        ]
        rows.extend(
            {
                "drug_id": drug, "disease_id": disease, "label": 0,
                "label_source": "sampled_unobserved_pair", "split": name,
            }
            for drug, disease in negative_pool_by_split[name]
        )
        frame = pd.DataFrame(rows).sample(
            frac=1, random_state=SEED
        ).reset_index(drop=True)
        if frame.duplicated(["drug_id", "disease_id"]).any():
            raise AssertionError(f"Duplicate pairs in {name}")
        frame.to_csv(SPLIT_DIR / f"{name}_pairs.csv", index=False)
        summary[name] = {
            "drugs": len(drug_groups[name]),
            "positive": int((frame["label"] == 1).sum()),
            "negative": int((frame["label"] == 0).sum()),
            "total": len(frame),
        }

    # Hold out every treatment edge associated with a validation/test drug.
    heldout_drugs = drug_groups["validation"] | drug_groups["test"]
    heldout_pairs = {
        pair for pair in positives if pair[0] in heldout_drugs
    }
    remove_mask = []
    for row in edges.itertuples(index=False):
        pair = (row.source_id, row.target_id)
        # For a held-out drug, remove every treatment label, including
        # labels on pairs excluded from supervision because they conflict
        # with a contraindication.  Otherwise a hidden label could remain.
        remove = (
            row.relation in TREATMENT_RELATIONS
            and (row.source_id in heldout_drugs or pair in conflicts)
        )
        if row.relation == CONTRAINDICATION and pair in conflicts:
            remove = True
        remove_mask.append(remove)

    training_edges = edges.loc[~pd.Series(remove_mask, index=edges.index)].copy()
    leaked = training_edges.loc[
        training_edges["relation"].isin(TREATMENT_RELATIONS)
        & training_edges["source_id"].isin(heldout_drugs)
    ]
    if not leaked.empty:
        raise AssertionError("Treatment labels leaked for held-out drugs")

    edge_ids = set(training_edges["edge_id"])
    training_evidence = evidence.loc[evidence["edge_id"].isin(edge_ids)].copy()
    if edge_ids - set(training_evidence["edge_id"]):
        raise AssertionError("Training graph has edges without evidence")

    training_edges.to_parquet(
        MODEL_DIR / "drug_cold_training_graph_edges.parquet", index=False
    )
    training_evidence.to_parquet(
        MODEL_DIR / "drug_cold_training_graph_evidence.parquet", index=False
    )

    metadata = {
        "dataset_version": "ReMedy-KG v1.0",
        "random_seed": SEED,
        "split_method": "drug-cold-start grouped split",
        "positive_definition": "indication or off-label use; conflicts excluded",
        "negative_definition": "sampled unobserved, non-contraindicated pair",
        "drug_group_counts": {name: len(group) for name, group in drug_groups.items()},
        "split_sizes": summary,
        "conflicting_pairs_excluded": len(conflicts),
        "heldout_drug_treatment_leakage": int(len(leaked)),
        "training_graph_edge_count": len(training_edges),
        "training_graph_evidence_rows": len(training_evidence),
    }
    (SPLIT_DIR / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("DRUG-COLD-START SPLITS CREATED")
    for name, values in summary.items():
        print(
            f"{name}: {values['drugs']} drugs, {values['positive']} positives, "
            f"{values['negative']} negatives"
        )
    print("Held-out drug treatment-label leakage:", len(leaked))
    print("Training graph edges:", len(training_edges))


if __name__ == "__main__":
    main()
