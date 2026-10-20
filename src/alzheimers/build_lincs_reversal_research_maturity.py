"""Assign research-maturity context to expanded Alzheimer LINCS candidates.

Retrieves Alzheimer-specific ClinicalTrials.gov and PubMed records for the small,
source-backed candidate set. Retrieval signals are preserved as research context;
benefit, harm, and suitability require source-cited outcome review.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


CTG = "https://clinicaltrials.gov/api/v2/studies"
PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def get_json(url: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    request = Request(url, headers={"User-Agent": "ReMedy academic evidence workflow"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(0.15)
            return payload
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def trial_records(drug: str, cache: Path) -> list[dict]:
    url = f"{CTG}?{urlencode({'query.intr': drug, 'query.cond': 'Alzheimer Disease', 'pageSize': 100, 'format': 'json'})}"
    studies = get_json(url, cache / "trials" / f"{drug.lower().replace(' ', '_')}.json").get("studies", [])
    rows = []
    for study in studies:
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        has_results = bool(study.get("hasResults") or study.get("resultsSection"))
        rows.append({
            "drug_name": drug,
            "nct_id": clean(ident.get("nctId")),
            "title": clean(ident.get("briefTitle")),
            "overall_status": clean(status.get("overallStatus")),
            "phase": " | ".join(design.get("phases", []) or []),
            "has_posted_results": has_results,
            "source_url": f"https://clinicaltrials.gov/study/{clean(ident.get('nctId'))}",
        })
    return rows


def pubmed_records(drug: str, cache: Path) -> list[dict]:
    term = f'("{drug}"[Title/Abstract]) AND ("Alzheimer Disease"[Title/Abstract])'
    url = f"{PUBMED}/esearch.fcgi?{urlencode({'db': 'pubmed', 'term': term, 'retmode': 'json', 'retmax': 10, 'sort': 'relevance'})}"
    payload = get_json(url, cache / "pubmed" / f"{drug.lower().replace(' ', '_')}.json")
    ids = payload.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    summary_url = f"{PUBMED}/esummary.fcgi?{urlencode({'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'json'})}"
    summary = get_json(summary_url, cache / "pubmed" / f"{drug.lower().replace(' ', '_')}_summary.json").get("result", {})
    return [{
        "drug_name": drug,
        "pmid": pmid,
        "article_title": clean(summary.get(pmid, {}).get("title")),
        "publication_year": clean(summary.get(pmid, {}).get("pubdate"))[:4],
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    } for pmid in ids]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.curated_dir / "lincs_reversal" / "off_panel_expansion"
    candidates = pd.read_csv(base / "phase4_lincs_off_panel_target_expansion.csv", dtype=str).fillna("")
    candidates = candidates.loc[
        candidates["expansion_status"].eq("OFF_PANEL_TARGETS_WITH_OPEN_TARGETS_AD_EVIDENCE")
    ].copy()
    cache = args.curated_dir / "raw_cache" / "lincs_research_maturity"
    all_trials: list[dict] = []
    all_papers: list[dict] = []
    for index, drug in enumerate(candidates["chembl_pref_name"].map(clean), start=1):
        all_trials.extend(trial_records(drug, cache))
        all_papers.extend(pubmed_records(drug, cache))
        print(f"Retrieved research context {index}/{len(candidates)}: {drug}", flush=True)

    trials = pd.DataFrame(all_trials)
    papers = pd.DataFrame(all_papers)
    trial_summary = (
        trials.groupby("drug_name", as_index=False).agg(
            alzheimer_trial_count=("nct_id", "count"),
            trial_statuses=("overall_status", lambda x: " | ".join(sorted(set(x)))),
            posted_result_trial_count=("has_posted_results", "sum"),
            trial_ids=("nct_id", lambda x: " | ".join(sorted(set(x)))),
        ) if not trials.empty else pd.DataFrame(columns=["drug_name", "alzheimer_trial_count", "trial_statuses", "posted_result_trial_count", "trial_ids"])
    )
    paper_summary = (
        papers.groupby("drug_name", as_index=False).agg(
            alzheimer_pubmed_record_count=("pmid", "count"),
            example_pmids=("pmid", lambda x: " | ".join(x[:3])),
            example_articles=("article_title", lambda x: " | ".join(x[:2])),
        ) if not papers.empty else pd.DataFrame(columns=["drug_name", "alzheimer_pubmed_record_count", "example_pmids", "example_articles"])
    )
    result = candidates.merge(trial_summary, left_on="chembl_pref_name", right_on="drug_name", how="left").drop(columns=["drug_name"], errors="ignore")
    result = result.merge(paper_summary, left_on="chembl_pref_name", right_on="drug_name", how="left").drop(columns=["drug_name"], errors="ignore").fillna("")
    for column in ["alzheimer_trial_count", "posted_result_trial_count", "alzheimer_pubmed_record_count"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    result["research_maturity_stage"] = result.apply(
        lambda row: "M3_POSTED_CLINICAL_OUTCOME_REVIEW_REQUIRED"
        if row["posted_result_trial_count"] > 0 else
        "M2_CLINICAL_RESEARCH_CONTEXT"
        if row["alzheimer_trial_count"] > 0 else
        "M1_LITERATURE_CONTEXT"
        if row["alzheimer_pubmed_record_count"] > 0 else
        "M0_NOVEL_REVERSAL_HYPOTHESIS",
        axis=1,
    )
    output = args.curated_dir / "lincs_reversal" / "research_maturity"
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output / "alzheimers_clinical_trial_records.csv", index=False)
    papers.to_csv(output / "alzheimers_pubmed_records.csv", index=False)
    result.to_csv(output / "alzheimers_lincs_reversal_research_maturity.csv", index=False)
    print(json.dumps({
        "expanded_candidates_screened": len(result),
        "research_maturity_counts": result["research_maturity_stage"].value_counts().to_dict(),
        "output": str(output / "alzheimers_lincs_reversal_research_maturity.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
