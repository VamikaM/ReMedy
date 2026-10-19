"""Build researcher-facing ReMedy evidence-stage cards without model scores.

The GNN is used upstream only to retrieve hypotheses. This script communicates
what evidence exists, what remains unresolved, and the next research action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def unique_join(values: pd.Series, limit: int = 5) -> str:
    seen = []
    for value in values:
        value = clean(value)
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return " | ".join(seen)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1" / "candidate_evidence_chain.csv",
    )
    parser.add_argument(
        "--known-use", type=Path,
        default=root / "reports" / "evidence_chain" / "chembl_known_use_screen.csv",
    )
    parser.add_argument(
        "--clinical", type=Path,
        default=root / "data" / "processed" / "clinical_trial_evidence.csv",
    )
    parser.add_argument(
        "--pubmed", type=Path,
        default=root / "data" / "processed" / "pubmed_evidence.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "reports" / "researcher_output" / "candidate_evidence_stage_cards.csv",
    )
    return parser.parse_args()


def stage(known: str, clinical_states: set[str], paper_count: int) -> tuple[str, str]:
    if known == "KNOWN_USE_EXACT_EFO_MATCH":
        return (
            "ESTABLISHED_USE_RECOVERED",
            "Treat as a known-use control, not a novel repurposing hypothesis.",
        )
    if known == "KNOWN_USE_TERM_MATCH_REVIEW":
        return (
            "POSSIBLE_KNOWN_USE_REQUIRES_ONTOLOGY_REVIEW",
            "Resolve the disease-term match before considering novelty.",
        )
    if clinical_states & {
        "POSTED_RESULTS_REVIEW_REQUIRED",
        "COMPLETED_NO_POSTED_RESULTS",
    }:
        return (
            "HUMAN_STUDY_OUTCOME_ADJUDICATION_REQUIRED",
            "Review the cited study's primary outcome, population, and adverse events.",
        )
    if clinical_states & {"NONCOMPLETED_TRIAL_REVIEW_REQUIRED"}:
        return (
            "HUMAN_STUDY_INCOMPLETE_REVIEW_REQUIRED",
            "Review why the registered study did not complete before further prioritisation.",
        )
    if clinical_states & {"TRIAL_IN_PROGRESS_OR_UNKNOWN"}:
        return (
            "HUMAN_STUDY_IN_PROGRESS_OR_UNRESOLVED",
            "Monitor the cited registry record; do not infer benefit.",
        )
    if paper_count:
        return (
            "MECHANISTIC_HYPOTHESIS_LITERATURE_RELEVANCE_REVIEW",
            "Screen cited literature for causal target, tissue exposure, phenotype, benefit, and harm evidence.",
        )
    return (
        "MECHANISTIC_HYPOTHESIS_EVIDENCE_RETRIEVAL_REQUIRED",
        "Retrieve and review causal-target, exposure, phenotype, and human evidence.",
    )


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates, dtype={"drug_id": str, "disease_id": str}).fillna("")
    known = pd.read_csv(args.known_use, dtype={"drug_id": str, "disease_id": str}).fillna("")
    clinical = pd.read_csv(args.clinical, dtype={"drug_id": str, "disease_id": str}).fillna("")
    pubmed = pd.read_csv(args.pubmed, dtype={"drug_id": str, "disease_id": str}).fillna("")

    for frame in (candidates, known, clinical, pubmed):
        for column in ("drug_id", "disease_id"):
            if column in frame:
                frame[column] = frame[column].map(clean)

    known_by_id = known.set_index("candidate_id").to_dict("index")
    clinical_by_pair = {
        pair: group for pair, group in clinical.groupby(["drug_id", "disease_id"], sort=False)
    }
    papers_by_pair = {
        pair: group for pair, group in pubmed.groupby(["drug_id", "disease_id"], sort=False)
    }

    cards = []
    for row in candidates.itertuples(index=False):
        pair = (clean(row.drug_id), clean(row.disease_id))
        known_record = known_by_id.get(row.candidate_id, {})
        known_state = clean(known_record.get("known_use_screen_status", "UNSCREENED"))
        trial_rows = clinical_by_pair.get(pair, clinical.iloc[0:0])
        paper_rows = papers_by_pair.get(pair, pubmed.iloc[0:0])
        clinical_states = {
            clean(state) for state in trial_rows.get(
                "clinical_evidence_state", pd.Series(dtype=str)
            )
        }
        stage_name, next_action = stage(known_state, clinical_states, len(paper_rows))
        cards.append({
            "candidate_id": row.candidate_id,
            "drug_name": row.drug_name,
            "disease_name": row.disease_name,
            "mechanistic_path": (
                f"{row.drug_name} --[{row.compatible_drug_actions}]--> "
                f"{row.compatible_proteins} --[genetic-risk direction]--> {row.disease_name}"
            ),
            "current_research_stage": stage_name,
            "known_use_screen": known_state,
            "clinical_trial_states": unique_join(
                trial_rows.get("clinical_evidence_state", pd.Series(dtype=str))
            ),
            "trial_ids": unique_join(trial_rows.get("nct_id", pd.Series(dtype=str))),
            "pubmed_record_count": int(len(paper_rows)),
            "example_pubmed_articles": unique_join(
                paper_rows.get("article_title", pd.Series(dtype=str)), limit=3
            ),
            "initial_safety_context": row.safety_status,
            "what_is_established": (
                "A direction-compatible drug-action to disease-target path was retrieved; "
                "this is a mechanistic signal, not therapeutic efficacy."
            ),
            "what_is_missing": (
                "Causal target-role review; disease-relevant tissue/exposure evidence; "
                "phenotype evidence; source-cited human outcome interpretation."
            ),
            "next_research_action": next_action,
            "source_links": " | ".join(filter(None, [
                unique_join(trial_rows.get("source_url", pd.Series(dtype=str))),
                unique_join(paper_rows.get("source_url", pd.Series(dtype=str))),
            ])),
        })

    result = pd.DataFrame(cards).sort_values(
        ["disease_name", "current_research_stage", "drug_name"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = {
        "candidate_cards": int(len(result)),
        "stage_counts": result["current_research_stage"].value_counts().to_dict(),
        "score_display_policy": "No GNN score is displayed in researcher-facing candidate cards.",
        "evidence_policy": "Stage reflects retrieved records and unresolved evidence, not a treatment claim.",
    }
    args.output.with_name("candidate_evidence_stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote researcher stage cards to {args.output}")


if __name__ == "__main__":
    main()
