"""Retrieve trial and literature context for mechanism-supported PrimeKG candidates.

Run for one selected disease at a time. Retrieved records establish research context;
they are not automatically interpreted as evidence of efficacy or safety.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease-name", required=True, help="Exact or distinctive disease name fragment.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--include-related-diseases", action="store_true",
        help="Also retrieve separately labelled related disease-name matches.",
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "model_v2")
    return parser.parse_args()


def get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "ReMedy research prototype (academic project contact: repository)"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def clinical_trials(drug: str, disease: str) -> list[dict[str, str]]:
    query = urlencode({"query.cond": disease, "query.intr": drug, "pageSize": 5, "format": "json"})
    payload = get_json(f"https://clinicaltrials.gov/api/v2/studies?{query}")
    trials = []
    for study in payload.get("studies", []):
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        nct = identification.get("nctId", "")
        trials.append({
            "nct_id": nct,
            "trial_title": identification.get("briefTitle", ""),
            "trial_status": status.get("overallStatus", ""),
            "trial_phases": " | ".join(design.get("phases", [])),
            "trial_has_posted_results": str(bool(study.get("hasResults", False))),
            "trial_url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
        })
    return trials


def pubmed(drug: str, disease: str) -> list[dict[str, str]]:
    term = f'("{drug}"[Title/Abstract]) AND ("{disease}"[Title/Abstract])'
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
        {"db": "pubmed", "term": term, "retmax": 5, "retmode": "json", "sort": "relevance"}
    )
    search = get_json(search_url).get("esearchresult", {})
    ids = search.get("idlist", [])
    if not ids:
        return []
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    )
    records = get_json(summary_url).get("result", {})
    articles = []
    for pmid in ids:
        record = records.get(pmid, {})
        articles.append({
            "pmid": pmid,
            "article_title": record.get("title", ""),
            "publication_date": record.get("pubdate", ""),
            "journal": record.get("fulljournalname", ""),
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return articles


def main() -> None:
    args = parse_args()
    cards = pd.read_csv(args.report_dir / "primekg_rgcn_researcher_cards.csv", dtype={"drug_id": str, "disease_id": str})
    supported = cards.loc[cards["mechanism_gate_status"].eq("MECHANISM_REVIEW_READY")].copy()
    matched = supported.loc[
        supported["disease_name"].str.contains(args.disease_name, case=False, na=False)
    ].copy()
    query_normalized = args.disease_name.casefold().strip()
    exact = matched.loc[matched["disease_name"].str.casefold().eq(query_normalized)].copy()
    related = matched.loc[~matched.index.isin(exact.index)].copy()
    if exact.empty and related.empty:
        suggestions = cards.loc[
            cards["disease_name"].str.contains(args.disease_name, case=False, na=False),
            "disease_name",
        ].drop_duplicates().head(10).tolist()
        raise RuntimeError(
            "No mechanism-supported candidates matched this disease. "
            f"Related disease names in the ranking: {suggestions}"
        )
    sort_columns = ["research_priority_rank", "model_score"]
    selected_parts = []
    if not exact.empty:
        selected_parts.append(
            exact.sort_values(sort_columns, ascending=[True, False]).head(args.top_k).assign(
                disease_match_scope="EXACT_DISEASE"
            )
        )
    else:
        selected_parts.append(
            related.sort_values(sort_columns, ascending=[True, False]).head(args.top_k).assign(
                disease_match_scope="RELATED_DISEASE_ONLY_NO_EXACT_MATCH"
            )
        )
        related = related.iloc[0:0]
    if args.include_related_diseases and not related.empty:
        selected_parts.append(
            related.sort_values(sort_columns, ascending=[True, False]).head(args.top_k).assign(
                disease_match_scope="RELATED_DISEASE_CONTEXT"
            )
        )
    selected = pd.concat(selected_parts, ignore_index=True)

    rows = []
    for index, candidate in enumerate(selected.itertuples(index=False), start=1):
        print(f"Retrieving {index}/{len(selected)}: {candidate.drug_name} -> {candidate.disease_name}", flush=True)
        trial_records, pubmed_records, errors = [], [], []
        try:
            trial_records = clinical_trials(candidate.drug_name, candidate.disease_name)
        except Exception as error:
            errors.append(f"ClinicalTrials.gov retrieval failed: {type(error).__name__}")
        time.sleep(0.35)
        try:
            pubmed_records = pubmed(candidate.drug_name, candidate.disease_name)
        except Exception as error:
            errors.append(f"PubMed retrieval failed: {type(error).__name__}")
        time.sleep(0.35)

        if trial_records:
            stage = "HUMAN_RESEARCH_CONTEXT_RETRIEVED"
            next_action = "Review linked trial endpoints and source-cite efficacy, harm, and population relevance."
        elif pubmed_records:
            stage = "LITERATURE_CONTEXT_RETRIEVED"
            next_action = "Screen retrieved papers for direct disease efficacy, model system, and target relevance."
        else:
            stage = "MECHANISM_ONLY_HYPOTHESIS"
            next_action = "Retrieve independent preclinical disease-model, exposure, and safety evidence."
        first_trial = trial_records[0] if trial_records else {}
        first_article = pubmed_records[0] if pubmed_records else {}
        rows.append({
            **candidate._asdict(),
            "research_context_stage": stage,
            "clinical_trial_count_retrieved": len(trial_records),
            "example_trial_id": first_trial.get("nct_id", ""),
            "example_trial_title": first_trial.get("trial_title", ""),
            "example_trial_url": first_trial.get("trial_url", ""),
            "clinical_trial_records": json.dumps(trial_records),
            "pubmed_record_count_retrieved": len(pubmed_records),
            "example_pmid": first_article.get("pmid", ""),
            "example_article_title": first_article.get("article_title", ""),
            "example_article_url": first_article.get("pubmed_url", ""),
            "pubmed_records": json.dumps(pubmed_records),
            "retrieval_errors": " | ".join(errors),
            "context_next_action": next_action,
        })

    output = pd.DataFrame(rows)
    safe_name = "".join(character if character.isalnum() else "_" for character in args.disease_name.lower()).strip("_")
    output_path = args.report_dir / f"primekg_research_context_{safe_name}.csv"
    output.to_csv(output_path, index=False)
    summary = {
        "disease_query": args.disease_name,
        "disease_match_scope_counts": output["disease_match_scope"].value_counts().to_dict(),
        "mechanism_supported_candidates_retrieved": int(len(output)),
        "research_context_stage_counts": output["research_context_stage"].value_counts().to_dict(),
        "policy": "Trial registration and literature retrieval provide research context. Endpoint/outcome assessment remains source-cited review.",
    }
    (args.report_dir / f"primekg_research_context_{safe_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote disease-specific research context to {output_path}")


if __name__ == "__main__":
    main()
