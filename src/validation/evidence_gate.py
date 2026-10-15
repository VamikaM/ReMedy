"""Apply ReMedy's conservative evidence gate to model-ranked drug-disease pairs.

The graph model is permitted to *generate* an association signal, but it is not
permitted to promote that signal as a researcher-facing repurposing hypothesis.
This script converts each scored pair into an auditable decision by checking:

* known indication / off-label-use / contraindication records;
* an explicit drug action against a disease-relevant protein in the opposite
  biological direction; and
* a disease-context safety-screen record.

Pairs with missing inputs are retained in an audit table with a clear status,
but are never written to the eligible shortlist.  The safety field records
whether a source-backed context screen is available; it is not a clinical
clearance decision.

Expected input tables (CSV or Parquet):

  ranked_pairs: drug_id, disease_id, hybrid_rgcn_score
  nodes: node_id, node_type, label
  edges: source_id, relation, target_id
  drug_mechanisms: drug_id, protein_id, action_type, source, evidence_id
  disease_target_effects: disease_id, protein_id, disease_effect, source,
                          evidence_id
  drug_safety_context: drug_id, disease_id, safety_status, source, evidence_id

Action direction: inhibitor/antagonist/blocker = -1; agonist/activator = +1.
Disease direction: harmful increase/gain = +1; harmful decrease/loss = -1.
A mechanism is direction-compatible only when the two signs multiply to -1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KNOWN_RELATIONS = {"indication", "off-label use", "contraindication"}
NEGATIVE_ACTION_TERMS = ("INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE", "SUPPRESSOR")
POSITIVE_ACTION_TERMS = ("AGONIST", "ACTIVATOR", "POSITIVE", "POTENTIATOR", "INDUCER")
POSITIVE_DISEASE_EFFECTS = {
    "INCREASED_ACTIVITY", "GAIN_OF_FUNCTION", "HARMFUL_GAIN", "RISK_GAIN",
    "OVERACTIVE", "UPREGULATED",
}
NEGATIVE_DISEASE_EFFECTS = {
    "DECREASED_ACTIVITY", "LOSS_OF_FUNCTION", "HARMFUL_LOSS", "PROTECTIVE_LOSS",
    "UNDERACTIVE", "DOWNREGULATED",
}
PASSING_SAFETY_STATUSES = {"NO_MATCHING_CONTEXT_FLAG"}
FLAGGED_SAFETY_STATUSES = {"FLAGGED", "CONTRAINDICATED", "HIGH_RISK"}


def read_table(path: Path, required: set[str], optional: bool = False) -> pd.DataFrame:
    if not path.exists():
        if optional:
            return pd.DataFrame(columns=sorted(required))
        raise FileNotFoundError(f"Required input is missing: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def upper(value: object) -> str:
    return text(value).upper().replace("-", "_").replace(" ", "_")


def action_sign(action_type: object) -> int:
    action = upper(action_type)
    if any(term in action for term in NEGATIVE_ACTION_TERMS):
        return -1
    if any(term in action for term in POSITIVE_ACTION_TERMS):
        return 1
    return 0


def disease_sign(effect: object) -> int:
    value = upper(effect)
    if value in POSITIVE_DISEASE_EFFECTS:
        return 1
    if value in NEGATIVE_DISEASE_EFFECTS:
        return -1
    return 0


def list_unique(values: pd.Series) -> str:
    values = [text(value) for value in values if text(value)]
    return " | ".join(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--mechanisms", type=Path, required=True)
    parser.add_argument("--disease-effects", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "reports" / "candidate_eligibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking = read_table(args.ranking, {"drug_id", "disease_id", "hybrid_rgcn_score"})
    nodes = read_table(args.nodes, {"node_id", "node_type", "label"})
    edges = read_table(args.edges, {"source_id", "relation", "target_id"})
    mechanisms = read_table(args.mechanisms,
                            {"drug_id", "protein_id", "action_type", "source", "evidence_id"})
    effects = read_table(args.disease_effects,
                         {"disease_id", "protein_id", "disease_effect", "source", "evidence_id"})
    safety = read_table(args.safety,
                        {"drug_id", "disease_id", "safety_status", "source", "evidence_id"})

    for frame, columns in [
        (ranking, ["drug_id", "disease_id"]),
        (nodes, ["node_id"]),
        (edges, ["source_id", "target_id"]),
        (mechanisms, ["drug_id", "protein_id"]),
        (effects, ["disease_id", "protein_id"]),
        (safety, ["drug_id", "disease_id"]),
    ]:
        for column in columns:
            frame[column] = frame[column].map(text)

    if ranking.duplicated(["drug_id", "disease_id"]).any():
        raise ValueError("ranking contains duplicate drug-disease pairs")

    labels = nodes.set_index("node_id")["label"].map(text).to_dict()
    relation_rows = edges.copy()
    relation_rows["relation_normalized"] = relation_rows["relation"].map(lambda x: text(x).lower())
    known = relation_rows[relation_rows["relation_normalized"].isin(KNOWN_RELATIONS)]
    known_by_pair = (known.groupby(["source_id", "target_id"])["relation_normalized"]
                      .agg(list_unique).to_dict())

    mechanisms = mechanisms.copy()
    effects = effects.copy()
    mechanisms["action_sign"] = mechanisms["action_type"].map(action_sign)
    effects["disease_sign"] = effects["disease_effect"].map(disease_sign)
    mechanism_by_drug = {drug: frame for drug, frame in mechanisms.groupby("drug_id")}
    effect_by_disease = {disease: frame for disease, frame in effects.groupby("disease_id")}
    empty_mechanisms = mechanisms.iloc[0:0].copy()
    empty_effects = effects.iloc[0:0].copy()
    safety_by_pair = {(row.drug_id, row.disease_id): row
                      for row in safety.itertuples(index=False)}

    records: list[dict[str, object]] = []
    for pair in ranking.itertuples(index=False):
        drug_id, disease_id = text(pair.drug_id), text(pair.disease_id)
        known_status = known_by_pair.get((drug_id, disease_id), "")
        drug_mechanisms = mechanism_by_drug.get(drug_id, empty_mechanisms)
        disease_effects = effect_by_disease.get(disease_id, empty_effects)
        linked = pd.merge(drug_mechanisms, disease_effects, on="protein_id", how="inner",
                          suffixes=("_drug", "_disease"))
        if not linked.empty:
            linked["is_compatible"] = linked["action_sign"] * linked["disease_sign"] == -1
            linked["is_conflicting"] = linked["action_sign"] * linked["disease_sign"] == 1
        compatible = linked[linked["is_compatible"]] if not linked.empty else linked
        conflicting = linked[linked["is_conflicting"]] if not linked.empty else linked

        safety_row = safety_by_pair.get((drug_id, disease_id))
        safety_status = upper(getattr(safety_row, "safety_status", "NOT_AVAILABLE"))
        safety_source = text(getattr(safety_row, "source", ""))
        safety_evidence = text(getattr(safety_row, "evidence_id", ""))

        data_missing = []
        if drug_mechanisms.empty:
            data_missing.append("drug_action")
        if disease_effects.empty:
            data_missing.append("disease_direction")
        if safety_row is None:
            data_missing.append("safety_context")

        if known_status:
            status = "CONTRAINDICATED" if "contraindication" in known_status else "KNOWN_USE_RECOVERED"
            reason = f"Existing graph relation: {known_status}"
        elif safety_status in FLAGGED_SAFETY_STATUSES:
            status, reason = "SAFETY_FLAGGED", f"Safety context status: {safety_status}"
        elif data_missing:
            status, reason = "EVIDENCE_INCOMPLETE", "Missing: " + ", ".join(data_missing)
        elif len(compatible):
            if safety_status in PASSING_SAFETY_STATUSES:
                status, reason = "EVIDENCE_SUPPORTED", "Direction-compatible mechanism and source-backed safety context"
            else:
                status, reason = "SAFETY_CONTEXT_UNRESOLVED", f"Safety context status: {safety_status}"
        elif len(conflicting):
            status, reason = "DIRECTION_CONFLICT", "Matched protein actions reinforce the disease direction"
        else:
            status, reason = "NO_SIGNED_MECHANISTIC_PATH", "No direction-compatible drug-protein-disease chain"

        records.append({
            "drug_id": drug_id,
            "drug_name": labels.get(drug_id, drug_id),
            "disease_id": disease_id,
            "disease_name": labels.get(disease_id, disease_id),
            "model_score": float(pair.hybrid_rgcn_score),
            "eligibility_status": status,
            "decision_reason": reason,
            "known_relation": known_status,
            "mechanism_records_for_drug": int(len(drug_mechanisms)),
            "disease_direction_records": int(len(disease_effects)),
            "shared_signed_mechanisms": int(len(linked)),
            "direction_compatible_mechanisms": int(len(compatible)),
            "direction_conflicting_mechanisms": int(len(conflicting)),
            "compatible_proteins": list_unique(compatible.get("protein_id", pd.Series(dtype=str))),
            "compatible_drug_actions": list_unique(compatible.get("action_type", pd.Series(dtype=str))),
            "compatible_disease_effects": list_unique(compatible.get("disease_effect", pd.Series(dtype=str))),
            "mechanism_sources": list_unique(compatible.get("source_drug", pd.Series(dtype=str))),
            "mechanism_evidence_ids": list_unique(compatible.get("evidence_id_drug", pd.Series(dtype=str))),
            "disease_effect_sources": list_unique(compatible.get("source_disease", pd.Series(dtype=str))),
            "disease_effect_evidence_ids": list_unique(compatible.get("evidence_id_disease", pd.Series(dtype=str))),
            "safety_status": safety_status,
            "safety_source": safety_source,
            "safety_evidence_id": safety_evidence,
        })

    audit = pd.DataFrame(records).sort_values(["model_score", "drug_id", "disease_id"], ascending=[False, True, True])
    eligible = audit[audit["eligibility_status"] == "EVIDENCE_SUPPORTED"].copy()
    eligible.insert(0, "disease_rank", eligible.groupby("disease_id").cumcount() + 1)
    audit.to_csv(args.output_dir / "candidate_gate_audit.csv", index=False)
    eligible.to_csv(args.output_dir / "evidence_supported_candidates.csv", index=False)
    summary = {
        "input_pairs": int(len(audit)),
        "eligible_pairs": int(len(eligible)),
        "status_counts": audit["eligibility_status"].value_counts().to_dict(),
        "policy": {
            "known_use": "withheld from novel-candidate shortlist",
            "mechanism": "requires direction-compatible drug-protein-disease chain",
            "safety": "requires source-backed NO_MATCHING_CONTEXT_FLAG status",
        },
    }
    (args.output_dir / "candidate_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

