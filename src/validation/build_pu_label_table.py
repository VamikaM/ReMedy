"""Build a frozen-split positive-unlabelled label table for ReMedy V3.

The existing train split contains known positive treatment links and sampled
unlabelled pairs. V3 must not silently convert all unlabelled pairs into
negative labels. This script keeps them explicitly UNLABELLED, then adds only
source-backed contraindications or reviewed negative human outcomes as reliable
negatives. Validation and test files are intentionally not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POSITIVE_VERDICTS = {"POSITIVE_HUMAN_EVIDENCE"}
NEGATIVE_VERDICTS = {
    "NEGATIVE_HUMAN_EVIDENCE",
    "DISEASE_RELEVANT_SAFETY_SIGNAL",
}


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def read_csv(path: Path, required: set[str], optional: bool = False) -> pd.DataFrame:
    if path is None or not path.exists():
        if optional:
            return pd.DataFrame(columns=sorted(required))
        raise FileNotFoundError(f"Required input is missing: {path}")
    frame = pd.read_csv(path, dtype={"drug_id": str, "disease_id": str})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    for column in ("drug_id", "disease_id"):
        frame[column] = frame[column].map(clean)
    return frame


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs", type=Path,
        default=root / "data" / "splits" / "train_pairs.csv",
    )
    parser.add_argument(
        "--nodes", type=Path,
        default=root / "data" / "processed" / "nodes.parquet",
    )
    parser.add_argument(
        "--edges", type=Path,
        default=root / "data" / "processed" / "edges.parquet",
    )
    parser.add_argument(
        "--adjudications", type=Path, default=None,
        help="Completed candidate_adjudication_template.csv; optional.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "model" / "v3_pu_train_labels.csv",
    )
    parser.add_argument(
        "--summary", type=Path,
        default=root / "reports" / "model_evaluation" / "v3_pu_label_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_csv(args.train_pairs, {"drug_id", "disease_id", "label"})
    nodes = pd.read_parquet(args.nodes)
    edges = pd.read_parquet(args.edges)

    for frame in (nodes, edges):
        for column in ("node_id", "source_id", "target_id"):
            if column in frame.columns:
                frame[column] = frame[column].map(clean)

    node_type = nodes.set_index("node_id")["node_type"].to_dict()
    labels = train[["drug_id", "disease_id", "label"]].copy()
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    labels["label_state"] = labels["label"].map({
        1: "CONFIRMED_POSITIVE",
        0: "UNLABELLED",
    })
    labels["training_target"] = labels["label"].map({1: 1, 0: pd.NA})
    labels["label_source"] = labels["label"].map({
        1: "frozen_known_treatment_split",
        0: "frozen_unlabelled_pair",
    })
    labels["source_citation"] = ""

    contraindications = edges.loc[
        edges["relation"].astype(str).str.lower().eq("contraindication"),
        ["source_id", "target_id"],
    ].copy()
    contraindications.columns = ["drug_id", "disease_id"]
    contraindications = contraindications.loc[
        contraindications["drug_id"].map(node_type).eq("drug")
        & contraindications["disease_id"].map(node_type).eq("disease")
    ].drop_duplicates()

    reliable_negative = contraindications.assign(
        label=0,
        label_state="RELIABLE_NEGATIVE",
        training_target=0,
        label_source="source_backed_contraindication",
        source_citation="knowledge_graph:contraindication",
    )

    adjudications = read_csv(args.adjudications, {
        "drug_id", "disease_id", "review_verdict", "source_citation",
    }, optional=True)
    reviewed_records = []
    if not adjudications.empty:
        adjudications["review_verdict"] = adjudications["review_verdict"].map(
            lambda value: clean(value).upper()
        )
        for row in adjudications.itertuples(index=False):
            if row.review_verdict in POSITIVE_VERDICTS:
                reviewed_records.append({
                    "drug_id": row.drug_id, "disease_id": row.disease_id,
                    "label": 1, "label_state": "CONFIRMED_POSITIVE",
                    "training_target": 1,
                    "label_source": "source_cited_human_outcome_review",
                    "source_citation": clean(row.source_citation),
                })
            elif row.review_verdict in NEGATIVE_VERDICTS:
                reviewed_records.append({
                    "drug_id": row.drug_id, "disease_id": row.disease_id,
                    "label": 0, "label_state": "RELIABLE_NEGATIVE",
                    "training_target": 0,
                    "label_source": "source_cited_human_outcome_review",
                    "source_citation": clean(row.source_citation),
                })

    if reviewed_records:
        reliable_negative = pd.concat(
            [reliable_negative, pd.DataFrame(
                [row for row in reviewed_records if row["label"] == 0]
            )],
            ignore_index=True,
        )
        reviewed_positive = pd.DataFrame(
            [row for row in reviewed_records if row["label"] == 1]
        )
        labels = pd.concat([labels, reviewed_positive], ignore_index=True)

    labels = pd.concat([labels, reliable_negative], ignore_index=True)
    labels = labels.sort_values(
        ["drug_id", "disease_id", "training_target"],
        ascending=[True, True, False],
        na_position="last",
    ).drop_duplicates(["drug_id", "disease_id"], keep="first")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    summary = {
        "frozen_train_pairs": int(len(train)),
        "v3_train_rows": int(len(labels)),
        "label_state_counts": labels["label_state"].value_counts().to_dict(),
        "supervised_positive_count": int(labels["training_target"].eq(1).sum()),
        "reliable_negative_count": int(labels["training_target"].eq(0).sum()),
        "unlabelled_count": int(labels["training_target"].isna().sum()),
        "test_split_modified": False,
        "validation_split_modified": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote V3 positive-unlabelled training labels to {args.output}")


if __name__ == "__main__":
    main()
