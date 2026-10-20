"""Build compact, source-linked target-causality review packets for ReMedy.

The initial graph gate uses an Open Targets genetic-risk direction.  This tool
does not convert that association into a causal or therapeutic conclusion. It
collects the exact evidence identifiers, direction fields, scores, and direct
Open Targets record URLs needed to review each distinct disease-protein target
once instead of separately for every drug candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def ordered_join(values: pd.Series, limit: int | None = None) -> str:
    out: list[str] = []
    for raw in values:
        value = clean(raw)
        if value and value not in out:
            out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return " | ".join(out)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-review",
        type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1" / "disease_target_review.csv",
    )
    parser.add_argument(
        "--effects",
        type=Path,
        default=root / "data" / "processed" / "disease_target_effects.csv",
    )
    parser.add_argument(
        "--mapping-audit",
        type=Path,
        default=root / "reports" / "evidence_gate" / "open_targets_id_mapping_audit.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "evidence_chain" / "target_causality_review_packets.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.target_review, dtype=str).fillna("")
    effects = pd.read_csv(args.effects, dtype=str).fillna("")
    mapping = (pd.read_csv(args.mapping_audit, dtype=str).fillna("")
               if args.mapping_audit.exists() else pd.DataFrame())

    needed = {"disease_id", "disease_name", "protein_id"}
    if missing := needed - set(targets.columns):
        raise ValueError(f"Target review file missing columns: {sorted(missing)}")
    if missing := {"disease_id", "protein_id", "evidence_id", "source"} - set(effects.columns):
        raise ValueError(f"Disease-effect file missing columns: {sorted(missing)}")

    for frame in (targets, effects, mapping):
        for column in ("disease_id", "protein_id", "remedy_id"):
            if column in frame.columns:
                frame[column] = frame[column].map(clean)

    # Prefer the explicit mapping audit when present. Recovered Kaggle artifacts
    # may omit that report, in which case the same IDs are available directly in
    # disease_target_effects.csv and can safely reconstruct the review links.
    if not mapping.empty and {"entity_type", "remedy_id", "open_targets_id"} <= set(mapping.columns):
        protein_map = (
            mapping.loc[mapping["entity_type"].eq("protein"), ["remedy_id", "open_targets_id"]]
            .drop_duplicates("remedy_id")
            .set_index("remedy_id")["open_targets_id"]
            .to_dict()
        )
        disease_map = (
            mapping.loc[mapping["entity_type"].eq("disease"), ["remedy_id", "open_targets_id"]]
            .drop_duplicates("remedy_id")
            .set_index("remedy_id")["open_targets_id"]
            .to_dict()
        )
    else:
        disease_map = (
            effects.loc[effects.get("open_targets_disease_id", "").map(clean).ne(""),
                        ["disease_id", "open_targets_disease_id"]]
            .drop_duplicates("disease_id")
            .set_index("disease_id")["open_targets_disease_id"]
            .to_dict()
        ) if "open_targets_disease_id" in effects.columns else {}
        protein_map = (
            effects.loc[effects.get("open_targets_target_id", "").map(clean).ne(""),
                        ["protein_id", "open_targets_target_id"]]
            .drop_duplicates("protein_id")
            .set_index("protein_id")["open_targets_target_id"]
            .to_dict()
        ) if "open_targets_target_id" in effects.columns else {}

    summary = (
        effects.groupby(["disease_id", "protein_id"], as_index=False)
        .agg(
            open_targets_sources=("source", ordered_join),
            open_targets_evidence_ids=("evidence_id", ordered_join),
            genetic_risk_directions=("disease_effect", ordered_join),
            direction_on_target=("direction_on_target", ordered_join),
            direction_on_trait=("direction_on_trait", ordered_join),
            max_evidence_score=("evidence_score", lambda series: pd.to_numeric(series, errors="coerce").max()),
        )
    )
    packets = targets.merge(summary, on=["disease_id", "protein_id"], how="left")
    packets["open_targets_disease_id"] = packets["disease_id"].map(disease_map).fillna("")
    packets["open_targets_target_id"] = packets["protein_id"].map(protein_map).fillna("")
    packets["open_targets_disease_url"] = packets["open_targets_disease_id"].map(
        lambda value: f"https://platform.opentargets.org/disease/{value}" if value else ""
    )
    packets["open_targets_target_url"] = packets["open_targets_target_id"].map(
        lambda value: f"https://platform.opentargets.org/target/{value}" if value else ""
    )
    packets["automated_conclusion"] = "NONE"
    packets["review_question"] = (
        "Is this target a causal driver, protective factor, biomarker, or unsupported "
        "for this disease in a relevant tissue/cell type? Does the cited genetic-risk "
        "direction justify the proposed pharmacological modulation?"
    )
    packets["required_reviewer_fields"] = (
        "target_role; desired_modulation; relevant_tissue_or_cell; causal_evidence_tier; "
        "target_review_verdict; source_citation"
    )
    packets = packets.sort_values(["disease_name", "protein_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    packets.to_csv(args.output, index=False)
    report = {
        "unique_disease_protein_targets": int(len(packets)),
        "diseases": int(packets["disease_id"].nunique()),
        "mapped_open_targets_diseases": int(packets["open_targets_disease_id"].ne("").sum()),
        "mapped_open_targets_targets": int(packets["open_targets_target_id"].ne("").sum()),
        "policy": "Genetic association direction is preserved as evidence context and is never converted into a causal or therapeutic conclusion automatically.",
    }
    args.output.with_name("target_causality_packet_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote target-causality packets to {args.output}")


if __name__ == "__main__":
    main()
