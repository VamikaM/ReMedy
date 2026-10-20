"""Create a conservative, researcher-facing disposition for ReMedy candidates.

The upstream graph model retrieves hypotheses; it does not establish efficacy.  This
script merges the evidence-chain review, ChEMBL indication screen, trial retrieval,
and PubMed retrieval into a *research disposition*.  A disposition is deliberately
not a prescription or an approval claim.

Only source-cited decisions in the evidence-chain workspace can classify a candidate
as negative, safety-excluded, positive-human-evidence, or preclinical-ready.
Everything else remains withheld or requires review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def join_unique(values: pd.Series, limit: int = 5) -> str:
    ordered: list[str] = []
    for raw in values:
        value = clean(raw)
        if value and value not in ordered:
            ordered.append(value)
        if len(ordered) >= limit:
            break
    return " | ".join(ordered)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1" / "candidate_evidence_chain.csv",
    )
    parser.add_argument(
        "--chain-status",
        type=Path,
        default=root / "reports" / "evidence_chain" / "candidate_evidence_chain_status.csv",
    )
    parser.add_argument(
        "--known-use",
        type=Path,
        default=root / "reports" / "evidence_chain" / "chembl_known_use_screen.csv",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=root / "data" / "processed" / "clinical_trial_evidence.csv",
    )
    parser.add_argument(
        "--pubmed",
        type=Path,
        default=root / "data" / "processed" / "pubmed_evidence.csv",
    )
    parser.add_argument(
        "--outcome-review",
        type=Path,
        default=root / "data" / "curation" / "evidence_chain_v1" / "phenotype_human_outcome_review.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "researcher_output" / "candidate_research_dispositions.csv",
    )
    return parser.parse_args()


def disposition(
    known_use: str,
    safety_context: str,
    chain_stage: str,
    human_verdict: str,
    safety_verdict: str,
    has_outcome_citation: bool,
    clinical_states: set[str],
    paper_count: int,
) -> tuple[str, str, str, str]:
    """Return disposition, shortlist eligibility, rationale, and next action."""
    if known_use == "KNOWN_USE_EXACT_EFO_MATCH":
        return (
            "KNOWN_USE_CONTROL_NOT_NOVEL",
            "EXCLUDE_FROM_NOVEL_SHORTLIST",
            "A source-backed indication match was retrieved; retain only as a known-use control.",
            "Use this pair for sanity checking, not as a novel repurposing result.",
        )
    if known_use == "KNOWN_USE_TERM_MATCH_REVIEW":
        return (
            "POSSIBLE_KNOWN_USE_ONTOLOGY_REVIEW_REQUIRED",
            "WITHHOLD_PENDING_ONTOLOGY_REVIEW",
            "A disease-term indication match was retrieved, but the ontology identity is not exact.",
            "Resolve the source disease term and identifier before claiming novelty or use.",
        )
    if known_use == "CHEMBL_RETRIEVAL_ERROR":
        return (
            "KNOWN_USE_SCREEN_RETRIEVAL_ERROR",
            "WITHHOLD_PENDING_SOURCE_RETRIEVAL",
            "The known-use source could not be retrieved; absence of a match cannot be interpreted.",
            "Retry the ChEMBL indication retrieval before considering novelty.",
        )

    # These two findings are only decisive when the review has a citation.
    if has_outcome_citation and safety_verdict == "EXCLUDE":
        return (
            "EXCLUDED_SOURCE_CITED_SAFETY_EVIDENCE",
            "EXCLUDE_FROM_SHORTLIST",
            "A reviewer recorded disease-relevant safety exclusion with a source citation.",
            "Retain the citation in the audit trail; do not promote this pair.",
        )
    if has_outcome_citation and human_verdict == "NEGATIVE":
        return (
            "EXCLUDED_SOURCE_CITED_NEGATIVE_HUMAN_EVIDENCE",
            "EXCLUDE_FROM_SHORTLIST",
            "A reviewer recorded a negative human outcome with a source citation.",
            "Retain the outcome citation; do not promote this pair.",
        )

    if safety_context == "FLAGGED":
        return (
            "WITHHELD_DISEASE_RELEVANT_SAFETY_REVIEW",
            "EXCLUDE_UNTIL_REVIEWED",
            "The automated label screen found a disease-context safety signal; this is not a safety clearance.",
            "Review the linked authoritative label and enter a source-cited safety verdict.",
        )

    if chain_stage == "HUMAN_EVIDENCE_IDENTIFIED":
        return (
            "HUMAN_BENEFIT_REPORTED_QUALITY_REVIEW_REQUIRED",
            "REVIEW_BEFORE_SHORTLISTING",
            "A source-cited positive human outcome was entered, but study quality and benefit-risk still require appraisal.",
            "Independently check population, endpoint, comparator, effect size, and adverse events.",
        )
    if chain_stage == "PRECLINICAL_REPURPOSING_CANDIDATE":
        return (
            "ADVANCE_TO_INDEPENDENT_PRECLINICAL_VALIDATION",
            "RESEARCH_SHORTLIST",
            "Source-cited causal target, exposure, and positive phenotype evidence are recorded; no positive human claim is implied.",
            "Replicate in a disease-relevant model and define translational endpoints.",
        )
    if chain_stage == "HUMAN_OUTCOME_ADJUDICATION_REQUIRED":
        return (
            "HUMAN_OUTCOME_ADJUDICATION_REQUIRED",
            "WITHHOLD_PENDING_REVIEW",
            "A relevant human study exists, but its outcome has not been source-adjudicated.",
            "Review the primary endpoint and adverse-event results before any promotion.",
        )
    if "POSTED_RESULTS_REVIEW_REQUIRED" in clinical_states:
        return (
            "POSTED_TRIAL_RESULT_REVIEW_REQUIRED",
            "WITHHOLD_PENDING_REVIEW",
            "A registered study has posted results, but no source-cited benefit/harm verdict has been entered.",
            "Read and cite the posted primary outcome before classifying the candidate.",
        )
    if clinical_states & {
        "COMPLETED_NO_POSTED_RESULTS",
        "NONCOMPLETED_TRIAL_REVIEW_REQUIRED",
        "TRIAL_IN_PROGRESS_OR_UNKNOWN",
    }:
        return (
            "HUMAN_RESEARCH_CONTEXT_REVIEW_REQUIRED",
            "WITHHOLD_PENDING_REVIEW",
            "A relevant registry record exists but does not establish therapeutic benefit.",
            "Review study completion reason, publications, outcomes, and adverse events.",
        )
    if paper_count:
        return (
            "LITERATURE_RELEVANCE_REVIEW_REQUIRED",
            "WITHHOLD_PENDING_REVIEW",
            "Literature was retrieved, but relevance and direction of effect have not been source-adjudicated.",
            "Screen primary studies for target role, tissue exposure, phenotype, benefit, and harm.",
        )
    return (
        "INSUFFICIENT_EVIDENCE_FOR_RESEARCH_SHORTLIST",
        "WITHHOLD_PENDING_EVIDENCE",
        "No source-cited causal, exposure, phenotype, or human-outcome evidence currently supports promotion.",
        "Retrieve disease-specific target, exposure, and phenotype evidence before reconsideration.",
    )


def main() -> None:
    args = parse_args()
    required = [args.candidates, args.chain_status, args.known_use, args.clinical, args.pubmed, args.outcome_review]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite file(s). Run the evidence-chain, known-use, clinical, and PubMed steps first:\n"
            + "\n".join(missing)
        )

    candidates = pd.read_csv(args.candidates, dtype=str).fillna("")
    chain = pd.read_csv(args.chain_status, dtype=str).fillna("")
    known = pd.read_csv(args.known_use, dtype=str).fillna("")
    clinical = pd.read_csv(args.clinical, dtype=str).fillna("")
    pubmed = pd.read_csv(args.pubmed, dtype=str).fillna("")
    outcomes = pd.read_csv(args.outcome_review, dtype=str).fillna("")

    for frame in (candidates, chain, known, clinical, pubmed, outcomes):
        for column in ("candidate_id", "drug_id", "disease_id"):
            if column in frame.columns:
                frame[column] = frame[column].map(clean)

    # Older known-use screen files did not include source_url.  Preserve their
    # evidence status and create a stable ChEMBL record link rather than failing
    # the disposition build.
    if "source_url" not in known.columns:
        known["source_url"] = known.get(
            "drug_id", pd.Series("", index=known.index)
        ).map(
            lambda drug_id: (
                "https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/"
                f"molecule_chembl_id:{clean(drug_id)}"
                if clean(drug_id) else ""
            )
        )

    # The workspace stores an initial placeholder stage; keep the evaluator's
    # post-review stage under an unambiguous name.
    chain_cols = ["candidate_id", "evidence_chain_stage", "target_gate_status", "exposure_gate_status"]
    reviewed_chain = chain[chain_cols].rename(columns={
        "evidence_chain_stage": "reviewed_evidence_chain_stage",
        "target_gate_status": "reviewed_target_gate_status",
        "exposure_gate_status": "reviewed_exposure_gate_status",
    })
    known_cols = ["candidate_id", "known_use_screen_status", "source_url"]
    outcome_cols = [
        "candidate_id",
        "preclinical_phenotype_verdict",
        "human_outcome_verdict",
        "disease_relevant_safety_verdict",
        "source_citation",
        "review_notes",
    ]
    result = candidates.merge(reviewed_chain, on="candidate_id", how="left")
    result = result.merge(known[known_cols], on="candidate_id", how="left", suffixes=("", "_known_use"))
    result = result.merge(outcomes[outcome_cols], on="candidate_id", how="left")
    result = result.fillna("")

    clinical_by_pair = {
        pair: group for pair, group in clinical.groupby(["drug_id", "disease_id"], sort=False)
    }
    pubmed_by_pair = {
        pair: group for pair, group in pubmed.groupby(["drug_id", "disease_id"], sort=False)
    }

    rows = []
    for row in result.itertuples(index=False):
        pair = (clean(row.drug_id), clean(row.disease_id))
        trial_rows = clinical_by_pair.get(pair, clinical.iloc[0:0])
        paper_rows = pubmed_by_pair.get(pair, pubmed.iloc[0:0])
        clinical_states = {
            clean(value) for value in trial_rows.get("clinical_evidence_state", pd.Series(dtype=str))
        }
        human = clean(row.human_outcome_verdict).upper()
        safety = clean(row.disease_relevant_safety_verdict).upper()
        cited = bool(clean(row.source_citation))
        final, eligibility, basis, action = disposition(
            clean(row.known_use_screen_status),
            clean(row.safety_status),
            clean(row.reviewed_evidence_chain_stage),
            human,
            safety,
            cited,
            clinical_states,
            len(paper_rows),
        )
        rows.append({
            "candidate_id": row.candidate_id,
            "drug_name": row.drug_name,
            "disease_name": row.disease_name,
            "research_disposition": final,
            "novel_shortlist_eligibility": eligibility,
            "disposition_basis": basis,
            "next_action": action,
            "known_use_screen": clean(row.known_use_screen_status),
            "mechanism_context_screen": "MECHANISM_AND_CONTEXT_SCREEN_PASSED",
            "target_gate_status": clean(row.reviewed_target_gate_status),
            "exposure_gate_status": clean(row.reviewed_exposure_gate_status),
            "preclinical_phenotype_verdict": clean(row.preclinical_phenotype_verdict) or "PENDING",
            "human_outcome_verdict": human or "PENDING",
            "disease_relevant_safety_verdict": safety or "PENDING",
            "outcome_review_citation": clean(row.source_citation),
            "clinical_trial_states": join_unique(
                trial_rows.get("clinical_evidence_state", pd.Series(dtype=str))
            ),
            "trial_ids": join_unique(trial_rows.get("nct_id", pd.Series(dtype=str))),
            "pubmed_record_count": int(len(paper_rows)),
            "example_pubmed_articles": join_unique(
                paper_rows.get("article_title", pd.Series(dtype=str)), limit=3
            ),
            "known_use_source": clean(row.source_url),
        })

    output = pd.DataFrame(rows).sort_values(
        ["novel_shortlist_eligibility", "disease_name", "drug_name"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    summary = {
        "candidate_count": int(len(output)),
        "research_disposition_counts": output["research_disposition"].value_counts().to_dict(),
        "shortlist_eligibility_counts": output["novel_shortlist_eligibility"].value_counts().to_dict(),
        "promotion_rule": (
            "Only source-cited human-positive or preclinical-positive evidence-chain verdicts can "
            "reach review/shortlist dispositions. Graph paths, trial registration, PubMed retrieval, "
            "and absence of a label match are never treated as efficacy evidence."
        ),
    }
    summary_path = args.output.with_name("candidate_research_disposition_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote research dispositions to {args.output}")


if __name__ == "__main__":
    main()
