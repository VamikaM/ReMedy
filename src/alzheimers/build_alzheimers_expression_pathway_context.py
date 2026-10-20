"""Build Reactome pathway context from the labelled Alzheimer expression signature.

This module submits FDR-significant frontal-cortex genes to Reactome's public
Analysis Service.  It also maps the curated Alzheimer target panel to Reactome
so the output can show where disease-expression pathways and the candidate
target space overlap.  It does not infer drug efficacy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


REACTOME_URL = (
    "https://reactome.org/AnalysisService/identifiers/projection/"
    "?pageSize=1000&page=1"
)


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def submit_identifiers(identifiers: list[str], label: str) -> dict:
    """Submit a source-cited identifier list to Reactome's public API."""
    payload = ("#" + label + "\n" + "\n".join(sorted(set(identifiers)))).encode("utf-8")
    request = Request(
        REACTOME_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "text/plain",
            "Accept": "application/json",
            "User-Agent": "ReMedy academic pathway-context workflow",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"Reactome analysis request failed after retries: {last_error}")


def pathways_to_frame(payload: dict, prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for pathway in payload.get("pathways", []):
        entities = pathway.get("entities") or {}
        rows.append(
            {
                "pathway_id": clean(pathway.get("stId")),
                "pathway_name": clean(pathway.get("name")),
                f"{prefix}_entities_found": entities.get("found"),
                f"{prefix}_entities_total": entities.get("total"),
                f"{prefix}_p_value": entities.get("pValue"),
                f"{prefix}_fdr": entities.get("fdr"),
                f"{prefix}_ratio": entities.get("ratio"),
                "species": clean(pathway.get("species")),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_dir = args.curated_dir / "expression"
    signature_path = expression_dir / "alzheimers_frontal_cortex_expression_signature.csv"
    targets_path = args.curated_dir / "target_panel" / "target_evidence_panel.csv"
    if not signature_path.exists():
        raise FileNotFoundError(
            f"Missing {signature_path}. Run build_alzheimers_expression_signature.py first."
        )
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing {targets_path}. Build the target panel first.")

    signature = pd.read_csv(signature_path, dtype={"gene_symbol": str}).fillna("")
    significant = signature.loc[
        pd.to_numeric(signature["fdr_bh"], errors="coerce").le(args.fdr_threshold),
        "gene_symbol",
    ].map(clean)
    significant_genes = sorted({gene for gene in significant if gene})
    if len(significant_genes) < 10:
        raise RuntimeError(
            f"Only {len(significant_genes)} FDR-significant genes found; "
            "pathway analysis needs a meaningful signature."
        )

    targets = pd.read_csv(targets_path, dtype=str).fillna("")
    panel_genes = sorted({clean(gene).upper() for gene in targets["protein_label"] if clean(gene)})
    expression_payload = submit_identifiers(significant_genes, "AD_frontal_cortex_FDR_significant_genes")
    panel_payload = submit_identifiers(panel_genes, "ReMedy_Alzheimer_target_panel")

    expression_pathways = pathways_to_frame(expression_payload, "expression")
    panel_pathways = pathways_to_frame(panel_payload, "target_panel")
    pathway_context = expression_pathways.merge(
        panel_pathways.drop(columns=["pathway_name", "species"], errors="ignore"),
        on="pathway_id",
        how="left",
    )
    pathway_context["target_panel_entities_found"] = pathway_context[
        "target_panel_entities_found"
    ].fillna(0).astype(int)
    pathway_context["context_status"] = pathway_context.apply(
        lambda row: (
            "EXPRESSION_PATHWAY_WITH_TARGET_PANEL_OVERLAP"
            if row["target_panel_entities_found"] > 0
            else "EXPRESSION_PATHWAY_WITHOUT_CURRENT_TARGET_PANEL_OVERLAP"
        ),
        axis=1,
    )
    pathway_context = pathway_context.sort_values(
        ["expression_fdr", "expression_p_value", "pathway_name"],
        na_position="last",
    ).reset_index(drop=True)

    output_dir = expression_dir / "pathway_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    pathway_context.to_csv(output_dir / "alzheimers_expression_reactome_pathways.csv", index=False)
    pd.DataFrame({"gene_symbol": significant_genes}).to_csv(
        output_dir / "submitted_fdr_significant_genes.csv", index=False
    )
    (output_dir / "reactome_submission_metadata.json").write_text(
        json.dumps(
            {
                "source": "Reactome Analysis Service",
                "source_url": "https://reactome.org/AnalysisService/",
                "significant_gene_count": len(significant_genes),
                "target_panel_gene_count": len(panel_genes),
                "fdr_threshold": args.fdr_threshold,
                "policy": (
                    "Pathway enrichment provides disease-context evidence. "
                    "It does not establish target causality, drug exposure, safety, or efficacy."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "submitted_fdr_significant_genes": len(significant_genes),
                "reactome_pathways_returned": len(pathway_context),
                "pathways_with_target_panel_overlap": int(
                    pathway_context["target_panel_entities_found"].gt(0).sum()
                ),
                "output": str(output_dir / "alzheimers_expression_reactome_pathways.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
