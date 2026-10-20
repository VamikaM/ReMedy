"""Build a consensus Alzheimer brain-expression signature from public GEO cohorts.

Cohort:
- GSE118553: human postmortem Alzheimer and control brain samples.

The script derives a frontal-cortex disease-expression signature from a GEO
cohort whose sample-level disease labels are present in the downloaded matrix.
It produces disease-expression context for evidence review; drug action is
handled in later, separately validated stages.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


COHORTS = {
    "GSE118553": {
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/"
            "GSE118553/matrix/GSE118553_series_matrix.txt.gz"
        ),
        "platform": "GPL10558",
        "tissue_filter": "tissue: frontal_cortex",
        "description": (
            "Human postmortem frontal-cortex expression cohort with "
            "Alzheimer disease and control sample labels."
        ),
    },
}


def platform_annotation_url(platform: str) -> str:
    # GEO groups platform files by the first platform-digit prefix, e.g.
    # GPL10558 lives under GPL10nnn.
    prefix = platform[:5] if len(platform) > 7 else platform[:4]
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}nnn/"
        f"{platform}/annot/{platform}.annot.gz"
    )


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def request_download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "ReMedy academic expression analysis"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=180) as response, destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def split_tsv(line: str) -> list[str]:
    return next(csv.reader([line.rstrip("\n")], delimiter="\t"))


def read_series_metadata(path: Path) -> tuple[list[str], list[str]]:
    """Collect all study-design fields that can legitimately carry phenotype labels.

    GEO studies vary: some put diagnosis in characteristics, while GSE33000
    stores its case/control label in title/source-name fields.  Reference and
    technical samples remain unclassified rather than being forced into a
    disease group.
    """
    sample_ids: list[str] = []
    phenotype_rows: list[list[str]] = []
    ignored_fields = {
        "!Sample_geo_accession",
        "!Sample_relation",
        "!Sample_supplementary_file_1",
        "!Sample_supplementary_file_2",
        "!Sample_supplementary_file_3",
    }
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("!series_matrix_table_begin"):
                break
            if not raw_line.startswith("!Sample_"):
                continue
            values = [clean(value) for value in split_tsv(raw_line)]
            if not values:
                continue
            field = values[0]
            if field == "!Sample_geo_accession":
                sample_ids = values[1:]
            elif field not in ignored_fields:
                # Studies are inconsistent about where diagnosis is recorded.
                # Retain every per-sample annotation, then classify only explicit
                # Alzheimer/control labels and leave all other samples unlabelled.
                phenotype_rows.append(values[1:])
    if not sample_ids:
        raise RuntimeError(f"No sample accessions found in {path.name}.")
    descriptions = [
        " | ".join(
            row[index] for row in phenotype_rows if index < len(row) and row[index]
        ).lower()
        for index in range(len(sample_ids))
    ]
    return sample_ids, descriptions


def classify_samples(sample_ids: list[str], descriptions: list[str]) -> tuple[list[str], list[str]]:
    cases: list[str] = []
    controls: list[str] = []
    for sample_id, description in zip(sample_ids, descriptions):
        is_ad = (
            "alzheimer" in description
            or bool(re.search(r"\blate onset ad\b|\bload\b|\bad\b", description))
        )
        is_hd = "huntington" in description
        is_control = (
            "non-demented" in description
            or "nondemented" in description
            or "control" in description
            or "normal" in description
        )
        if is_ad and not is_hd:
            cases.append(sample_id)
        elif is_control and not is_hd and not is_ad:
            controls.append(sample_id)
    return cases, controls


def read_matrix(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("!series_matrix_table_begin"):
                break
        matrix = pd.read_csv(handle, sep="\t", dtype=str)
    id_column = matrix.columns[0]
    matrix = matrix.loc[~matrix[id_column].astype(str).str.startswith("!series_matrix_table_end")]
    matrix = matrix.rename(columns={id_column: "probe_id"})
    matrix["probe_id"] = matrix["probe_id"].map(clean)
    return matrix


def read_platform_annotation(path: Path) -> pd.DataFrame:
    # GEO platform annotations begin with metadata lines, then an explicit table.
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("!platform_table_begin"):
                break
        else:
            raise RuntimeError(f"Could not find !platform_table_begin in {path.name}.")
        annotation = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)
    first_column = annotation.columns[0]
    annotation = annotation.loc[
        ~annotation[first_column].astype(str).str.startswith("!platform_table_end")
    ]
    annotation.columns = [clean(column) for column in annotation.columns]
    return annotation.fillna("")


def find_column(columns: list[str], candidates: list[str]) -> str:
    normalized = {re.sub(r"[^a-z0-9]+", "", column.lower()): column for column in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    for column in columns:
        lowered = column.lower()
        if "gene symbol" in lowered or lowered == "symbol":
            return column
    raise RuntimeError(f"Could not locate any of {candidates} in platform annotation columns.")


def gene_symbol(value: object) -> str:
    symbol = clean(value)
    if not symbol or symbol in {"---", "NA", "N/A"}:
        return ""
    symbol = re.split(r"\s*(?:///|//|;|,)\s*", symbol)[0].strip()
    return symbol.upper() if re.fullmatch(r"[A-Za-z0-9._-]+", symbol) else ""


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    output = np.full(len(pvalues), np.nan)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return output
    values = pvalues[valid]
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    adjusted = values[order] * len(values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty(len(values))
    restored[order] = adjusted
    output[valid] = restored
    return output


def differential_expression(
    cohort: str,
    matrix: pd.DataFrame,
    cases: list[str],
    controls: list[str],
    probe_to_gene: dict[str, str],
) -> pd.DataFrame:
    sample_columns = [column for column in cases + controls if column in matrix.columns]
    cases = [sample for sample in cases if sample in sample_columns]
    controls = [sample for sample in controls if sample in sample_columns]
    if len(cases) < 3 or len(controls) < 3:
        raise RuntimeError(f"{cohort}: insufficient classified samples (cases={len(cases)}, controls={len(controls)}).")

    values = matrix[["probe_id", *sample_columns]].copy()
    values["gene_symbol"] = values["probe_id"].map(probe_to_gene).fillna("")
    values = values.loc[values["gene_symbol"].ne("")].copy()
    values[sample_columns] = values[sample_columns].apply(pd.to_numeric, errors="coerce")
    values = values.dropna(subset=sample_columns, how="all")
    grouped = values.groupby("gene_symbol", as_index=True)
    expression = grouped[sample_columns].mean()
    probe_counts = grouped.size()

    case_matrix = expression[cases].to_numpy(float)
    control_matrix = expression[controls].to_numpy(float)
    test = ttest_ind(case_matrix, control_matrix, axis=1, equal_var=False, nan_policy="omit")
    effect = np.nanmean(case_matrix, axis=1) - np.nanmean(control_matrix, axis=1)
    result = pd.DataFrame(
        {
            "gene_symbol": expression.index,
            "cohort": cohort,
            "mean_case_expression": np.nanmean(case_matrix, axis=1),
            "mean_control_expression": np.nanmean(control_matrix, axis=1),
            "log_expression_difference": effect,
            "t_statistic": test.statistic,
            "p_value": test.pvalue,
            "probe_count": probe_counts.reindex(expression.index).to_numpy(),
            "n_cases": len(cases),
            "n_controls": len(controls),
        }
    )
    result["fdr_bh"] = benjamini_hochberg(result["p_value"].to_numpy(float))
    result["expression_direction"] = np.where(
        result["log_expression_difference"].gt(0), "INCREASED_IN_AD", "DECREASED_IN_AD"
    )
    return result.sort_values(["fdr_bh", "p_value", "gene_symbol"]).reset_index(drop=True)


def consensus_signature(results: list[pd.DataFrame]) -> pd.DataFrame:
    left = results[0].set_index("gene_symbol")
    right = results[1].set_index("gene_symbol")
    shared = left.index.intersection(right.index)
    left = left.loc[shared]
    right = right.loc[shared]
    same_direction = np.sign(left["log_expression_difference"]) == np.sign(right["log_expression_difference"])
    strict = same_direction & left["fdr_bh"].le(0.05) & right["fdr_bh"].le(0.05)
    supporting = same_direction & (left["fdr_bh"].le(0.05) | right["fdr_bh"].le(0.05))
    category = np.where(
        strict,
        "STRICT_TWO_COHORT_CONSENSUS",
        np.where(supporting, "DIRECTIONALLY_CONSISTENT_SUPPORTING_SIGNAL", "NO_TWO_COHORT_CONSENSUS"),
    )
    output = pd.DataFrame(
        {
            "gene_symbol": shared,
            "gse33000_effect": left["log_expression_difference"].to_numpy(),
            "gse33000_fdr": left["fdr_bh"].to_numpy(),
            "gse44770_effect": right["log_expression_difference"].to_numpy(),
            "gse44770_fdr": right["fdr_bh"].to_numpy(),
            "mean_effect": (left["log_expression_difference"].to_numpy() + right["log_expression_difference"].to_numpy()) / 2,
            "same_direction": same_direction,
            "expression_evidence_category": category,
        }
    )
    output["consensus_direction"] = np.where(
        output["mean_effect"].gt(0), "INCREASED_IN_AD", "DECREASED_IN_AD"
    )
    return output.sort_values(
        ["expression_evidence_category", "mean_effect"],
        ascending=[True, False],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    curated = root / "data" / "curated" / "alzheimers_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-dir", type=Path, default=curated)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "raw" / "alzheimers_expression",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.curated_dir / "expression"
    output_dir.mkdir(parents=True, exist_ok=True)

    accession, config = next(iter(COHORTS.items()))
    platform = clean(config["platform"])
    platform_path = args.cache_dir / f"{platform}.annot.gz"
    request_download(platform_annotation_url(platform), platform_path)
    annotation = read_platform_annotation(platform_path)
    annotation_probe_column = find_column(list(annotation.columns), ["ID", "ID_REF", "Probe ID"])
    annotation_symbol_column = find_column(
        list(annotation.columns), ["Gene Symbol", "Gene symbol", "Symbol"]
    )
    probe_to_gene = {
        clean(probe): gene_symbol(symbol)
        for probe, symbol in zip(annotation[annotation_probe_column], annotation[annotation_symbol_column])
        if clean(probe) and gene_symbol(symbol)
    }
    if not probe_to_gene:
        raise RuntimeError("Platform annotation did not yield any probe-to-gene mappings.")

    matrix_path = args.cache_dir / f"{accession}_series_matrix.txt.gz"
    request_download(clean(config["matrix_url"]), matrix_path)
    sample_ids, descriptions = read_series_metadata(matrix_path)
    cases, controls = classify_samples(sample_ids, descriptions)

    sample_text = dict(zip(sample_ids, descriptions))
    tissue_filter = clean(config["tissue_filter"]).lower()
    cases = [sample for sample in cases if tissue_filter in sample_text[sample]]
    controls = [sample for sample in controls if tissue_filter in sample_text[sample]]
    matrix = read_matrix(matrix_path)
    result = differential_expression(accession, matrix, cases, controls, probe_to_gene)
    result.to_csv(output_dir / f"{accession}_frontal_cortex_differential_expression.csv", index=False)

    signature = result[
        [
            "gene_symbol",
            "cohort",
            "mean_case_expression",
            "mean_control_expression",
            "log_expression_difference",
            "t_statistic",
            "p_value",
            "fdr_bh",
            "probe_count",
            "n_cases",
            "n_controls",
            "expression_direction",
        ]
    ].copy()
    signature["expression_evidence_category"] = np.where(
        signature["fdr_bh"].le(0.05),
        "FDR_SIGNIFICANT_FRONTAL_CORTEX_SIGNAL",
        "FRONTAL_CORTEX_CONTEXT_ONLY",
    )
    signature = signature.rename(columns={"expression_direction": "consensus_direction"})
    signature.to_csv(output_dir / "alzheimers_frontal_cortex_expression_signature.csv", index=False)

    targets_path = args.curated_dir / "target_panel" / "target_evidence_panel.csv"
    if targets_path.exists():
        targets = pd.read_csv(targets_path, dtype=str).fillna("")
        targets["target_symbol"] = targets["protein_label"].map(lambda value: clean(value).upper())
        target_expression = targets.merge(
            signature,
            left_on="target_symbol",
            right_on="gene_symbol",
            how="left",
        )
        target_expression["expression_evidence_category"] = target_expression[
            "expression_evidence_category"
        ].fillna("NO_FRONTAL_CORTEX_SIGNATURE_MATCH")
        target_expression.to_csv(output_dir / "target_expression_context.csv", index=False)

    significant_count = int(
        signature["expression_evidence_category"].eq(
            "FDR_SIGNIFICANT_FRONTAL_CORTEX_SIGNAL"
        ).sum()
    )
    manifest = {
        "cohort": accession,
        "cohort_description": config["description"],
        "tissue_filter": config["tissue_filter"],
        "classified_alzheimer_cases": len(cases),
        "classified_controls": len(controls),
        "platform_annotation": platform,
        "probe_to_gene_mappings": int(len(probe_to_gene)),
        "mapped_genes_tested": int(len(result)),
        "fdr_significant_frontal_cortex_genes": significant_count,
        "policy": (
            "This file records disease-expression context in labelled postmortem "
            "frontal-cortex samples. It is used to prioritize evidence review and "
            "does not alone establish target causality or a therapeutic effect."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote Alzheimer frontal-cortex expression outputs to {output_dir}")


if __name__ == "__main__":
    main()
