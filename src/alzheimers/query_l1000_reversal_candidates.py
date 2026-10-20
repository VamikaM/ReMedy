"""Retrieve auditable LINCS reversal signals for the Alzheimer expression signature.

The workflow submits the strongest Alzheimer-up and Alzheimer-down frontal-cortex
genes to L1000CDS2 in reverse mode. Returned perturbations are joined to the
curated ChEMBL Phase-4 Alzheimer target catalogue by conservative exact-name
normalisation. A reversal signal is a research hypothesis, never efficacy.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


L1000CDS2_URL = "https://maayanlab.cloud/L1000CDS2/query"


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalized_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", clean(value).upper())


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ReMedy academic drug-repurposing workflow",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"L1000CDS2 request failed after retries: {last_error}")


def joined_unique(values: pd.Series) -> str:
    return " | ".join(sorted({clean(value) for value in values if clean(value)}))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument("--genes-per-direction", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expression_dir = args.curated_dir / "expression"
    signature_path = expression_dir / "alzheimers_frontal_cortex_expression_signature.csv"
    catalogue_dir = args.curated_dir / "expanded_sources" / "chembl"
    molecules_path = catalogue_dir / "approved_molecule_registry.csv"
    mechanisms_path = catalogue_dir / "approved_drug_target_catalog.csv"
    for path in [signature_path, molecules_path, mechanisms_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    signature = pd.read_csv(signature_path, dtype={"gene_symbol": str}).fillna("")
    signature["effect"] = pd.to_numeric(signature["log_expression_difference"], errors="coerce")
    signature["fdr"] = pd.to_numeric(signature["fdr_bh"], errors="coerce")
    signature = signature.dropna(subset=["effect"]).copy()
    signature = signature.loc[signature["gene_symbol"].map(clean).ne("")]
    signature = signature.sort_values(["effect", "fdr"], ascending=[False, True])

    n = args.genes_per_direction
    up_genes = signature.head(n)["gene_symbol"].map(lambda value: clean(value).upper()).tolist()
    down_genes = signature.tail(n).sort_values("effect")["gene_symbol"].map(
        lambda value: clean(value).upper()
    ).tolist()
    if len(set(up_genes)) < 25 or len(set(down_genes)) < 25:
        raise RuntimeError("Insufficient ranked genes for a LINCS reversal query.")

    payload = {
        "data": {"upGenes": up_genes, "dnGenes": down_genes},
        "config": {
            "aggravate": False,
            "searchMethod": "geneSet",
            "share": False,
            "combination": False,
            "db-version": "latest",
        },
        "meta": [
            {"key": "project", "value": "ReMedy"},
            {"key": "query", "value": "Alzheimer frontal-cortex signature reversal"},
        ],
    }
    response = post_json(L1000CDS2_URL, payload)
    raw = pd.DataFrame(response.get("topMeta", []))
    if raw.empty:
        raise RuntimeError("L1000CDS2 returned no perturbation signatures.")
    raw.insert(0, "l1000_reverse_rank", range(1, len(raw) + 1))
    raw["pert_desc"] = raw.get("pert_desc", pd.Series("", index=raw.index)).map(clean)
    raw["normalized_drug_name"] = raw["pert_desc"].map(normalized_name)
    raw["l1000_source"] = "LINCS L1000CDS2"
    raw["l1000_query_mode"] = "reverse"
    raw["l1000_query_url"] = L1000CDS2_URL

    molecules = pd.read_csv(molecules_path, dtype=str).fillna("")
    mechanisms = pd.read_csv(mechanisms_path, dtype=str).fillna("")
    molecules["normalized_drug_name"] = molecules["pref_name"].map(normalized_name)
    mechanisms = mechanisms.merge(
        molecules[["molecule_chembl_id", "pref_name", "normalized_drug_name"]],
        left_on="drug_id",
        right_on="molecule_chembl_id",
        how="left",
    )
    context = (
        mechanisms.groupby("normalized_drug_name", as_index=False)
        .agg(
            chembl_drug_ids=("drug_id", joined_unique),
            curated_drug_names=("pref_name", joined_unique),
            source_backed_target_count=("protein_label", lambda x: len({clean(v) for v in x if clean(v)})),
            source_backed_targets=("protein_label", joined_unique),
            source_backed_actions=("action_type", joined_unique),
            source_backed_mechanisms=("mechanism_of_action", joined_unique),
        )
    )

    matched = raw.merge(context, on="normalized_drug_name", how="left")
    matched["has_curated_alzheimer_target_context"] = matched["chembl_drug_ids"].notna()
    matched["candidate_status"] = matched["has_curated_alzheimer_target_context"].map(
        {
            True: "LINCS_REVERSAL_SIGNAL_WITH_CURATED_TARGET_CONTEXT",
            False: "LINCS_REVERSAL_SIGNAL_WITHOUT_CURATED_TARGET_CONTEXT",
        }
    )
    aggregate = (
        matched.sort_values("l1000_reverse_rank")
        .groupby(["normalized_drug_name", "pert_desc"], as_index=False, dropna=False)
        .agg(
            best_l1000_reverse_rank=("l1000_reverse_rank", "min"),
            l1000_experimental_conditions=("sig_id", "count"),
            l1000_cell_lines=("cell_id", joined_unique),
            l1000_doses=("pert_dose", joined_unique),
            l1000_timepoints=("pert_time", joined_unique),
            l1000_signature_ids=("sig_id", joined_unique),
            l1000_scores=("score", joined_unique),
            chembl_drug_ids=("chembl_drug_ids", joined_unique),
            curated_drug_names=("curated_drug_names", joined_unique),
            source_backed_target_count=("source_backed_target_count", "max"),
            source_backed_targets=("source_backed_targets", joined_unique),
            source_backed_actions=("source_backed_actions", joined_unique),
            source_backed_mechanisms=("source_backed_mechanisms", joined_unique),
            candidate_status=("candidate_status", "first"),
            l1000_source=("l1000_source", "first"),
            l1000_query_mode=("l1000_query_mode", "first"),
            l1000_query_url=("l1000_query_url", "first"),
        )
        .sort_values("best_l1000_reverse_rank")
        .reset_index(drop=True)
    )
    aggregate["has_curated_alzheimer_target_context"] = aggregate["chembl_drug_ids"].map(clean).ne("")

    output_dir = args.curated_dir / "lincs_reversal"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "l1000_raw_reversal_experiments.csv", index=False)
    aggregate.to_csv(output_dir / "l1000_reversal_candidate_screen.csv", index=False)
    (output_dir / "l1000_reversal_query.json").write_text(
        json.dumps(
            {
                "query_payload": payload,
                "response_share_id": response.get("shareId"),
                "input_policy": (
                    "Top and bottom ranked genes from a labelled Alzheimer frontal-cortex "
                    "signature. L1000 retrieval is run in reverse mode."
                ),
                "interpretation_policy": (
                    "A LINCS reversal signal is a cell-line perturbation match. It requires "
                    "disease-target, exposure, safety, and outcome evidence review before "
                    "any research hypothesis can be promoted."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_up_genes": len(up_genes),
                "input_down_genes": len(down_genes),
                "l1000_reversal_experiments_returned": len(raw),
                "unique_reversal_perturbations": len(aggregate),
                "perturbations_with_curated_alzheimer_target_context": int(
                    aggregate["has_curated_alzheimer_target_context"].sum()
                ),
                "output": str(output_dir / "l1000_reversal_candidate_screen.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
