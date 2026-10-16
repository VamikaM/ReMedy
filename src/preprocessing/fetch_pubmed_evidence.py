"""Retrieve PubMed citations and abstracts for a bounded ReMedy candidate batch.

The script preserves literature records for review; it does not infer therapeutic
benefit from an abstract.  PubMed is queried only after the graph and mechanism
filters have produced a small evidence-retrieval batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def request_json(url: str, params: dict) -> dict:
    with urlopen(f"{url}?{urlencode(params)}", timeout=30) as response:
        return json.load(response)


def request_text(url: str, params: dict) -> str:
    with urlopen(f"{url}?{urlencode(params)}", timeout=30) as response:
        return response.read().decode("utf-8")


def text_at(element: ET.Element, path: str) -> str:
    found = element.find(path)
    return clean("".join(found.itertext())) if found is not None else ""


def parse_articles(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = text_at(article, "./MedlineCitation/PMID")
        title = text_at(article, "./MedlineCitation/Article/ArticleTitle")
        abstract_parts = []
        for section in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
            label = section.attrib.get("Label", "").strip()
            body = clean("".join(section.itertext()))
            abstract_parts.append(f"{label}: {body}" if label else body)
        publication_types = [
            clean("".join(item.itertext()))
            for item in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
        ]
        journal = text_at(article, "./MedlineCitation/Article/Journal/Title")
        year = (
            text_at(article, "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
            or text_at(article, "./MedlineCitation/Article/ArticleDate/Year")
            or text_at(article, "./MedlineCitation/DateCompleted/Year")
        )
        records.append({
            "pmid": pmid,
            "article_title": title,
            "abstract": " ".join(part for part in abstract_parts if part),
            "journal": journal,
            "publication_year": year,
            "publication_types": " | ".join(item for item in publication_types if item),
            "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        })
    return records


def fetch_pair(
    drug_name: str, disease_name: str, cache_path: Path, max_articles: int
) -> list[dict]:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["records"]
    query = f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]'
    try:
        search = request_json(ESEARCH, {
            "db": "pubmed", "term": query, "retmode": "json", "retmax": max_articles,
            "tool": "ReMedy",
        })
        ids = search.get("esearchresult", {}).get("idlist", [])
        time.sleep(0.35)
        records = []
        if ids:
            xml_text = request_text(EFETCH, {
                "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "ReMedy",
            })
            records = parse_articles(xml_text)
            time.sleep(0.35)
        payload = {"records": records, "query": query}
    except Exception as error:
        payload = {"records": [], "query": query, "_retrieval_error": str(error)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["records"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="Bounded candidate CSV containing drug_id and disease_id.")
    parser.add_argument("--nodes", type=Path,
                        default=root / "data/processed/nodes.parquet")
    parser.add_argument("--max-articles-per-pair", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/pubmed_evidence.csv")
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "data/raw/pubmed_evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs, dtype={"drug_id": str, "disease_id": str})
    missing = {"drug_id", "disease_id"} - set(pairs.columns)
    if missing:
        raise ValueError(f"{args.pairs.name} is missing columns: {sorted(missing)}")
    pairs = pairs.drop_duplicates(["drug_id", "disease_id"])
    if args.max_articles_per_pair < 1:
        raise ValueError("--max-articles-per-pair must be at least 1")

    nodes = pd.read_parquet(args.nodes)
    labels = nodes.set_index("node_id")["label"].astype(str).to_dict()
    records: list[dict] = []
    total = len(pairs)
    for index, pair in enumerate(pairs.itertuples(index=False), start=1):
        drug_name, disease_name = labels.get(pair.drug_id, ""), labels.get(pair.disease_id, "")
        if index == 1 or index % 10 == 0 or index == total:
            print(f"PubMed retrieval {index}/{total}: {drug_name} -> {disease_name}", flush=True)
        key = hashlib.sha256(
            f"{pair.drug_id}|{pair.disease_id}".encode("utf-8")
        ).hexdigest()[:20]
        articles = fetch_pair(
            drug_name, disease_name, args.cache_dir / f"{key}.json",
            args.max_articles_per_pair,
        ) if drug_name and disease_name else []
        if articles:
            for article in articles:
                records.append({
                    "drug_id": pair.drug_id, "drug_name": drug_name,
                    "disease_id": pair.disease_id, "disease_name": disease_name,
                    "evidence_state": "LITERATURE_RETRIEVED",
                    "source": "PubMed", "evidence_id": f"PMID:{article['pmid']}",
                    **article,
                })
        else:
            records.append({
                "drug_id": pair.drug_id, "drug_name": drug_name,
                "disease_id": pair.disease_id, "disease_name": disease_name,
                "evidence_state": "NO_MATCHING_PUBMED_RECORD",
                "source": "PubMed", "evidence_id": "", "pmid": "",
                "article_title": "", "abstract": "", "journal": "",
                "publication_year": "", "publication_types": "", "source_url": "",
            })

    evidence = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.output, index=False)
    print(evidence["evidence_state"].value_counts().to_string())
    print(f"Wrote {len(evidence):,} PubMed evidence records to {args.output}")


if __name__ == "__main__":
    main()
