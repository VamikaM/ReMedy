# ReMedy evidence-gated candidate workflow

## Purpose

The Hybrid R-GCN generates an internal association signal. It does not decide that
a drug should be shown as a repurposing hypothesis. The evidence gate converts
every score into an auditable status and writes a shortlist only for pairs that
meet all required checks.

The workflow deliberately separates a **model signal** from an
**evidence-supported candidate**.

## Required source-backed inputs

| Input | Required fields | Source and purpose |
|---|---|---|
| `drug_mechanisms.csv` | `drug_id, protein_id, action_type, source, evidence_id` | ChEMBL mechanism records: identifies whether a drug inhibits, activates, antagonises or agonises a protein. |
| `disease_target_effects.csv` | `disease_id, protein_id, disease_effect, source, evidence_id` | Open Targets direction-of-effect evidence, mapped to ReMedy protein and disease IDs. Accepted values are `INCREASED_ACTIVITY`, `GAIN_OF_FUNCTION`, `HARMFUL_GAIN`, `RISK_GAIN`, `OVERACTIVE`, `UPREGULATED`, `DECREASED_ACTIVITY`, `LOSS_OF_FUNCTION`, `HARMFUL_LOSS`, `PROTECTIVE_LOSS`, `UNDERACTIVE`, or `DOWNREGULATED`. |
| `drug_safety_context.csv` | `drug_id, disease_id, safety_status, source, evidence_id` | Cached openFDA label screen. `FLAGGED` blocks a pair; `NO_MATCHING_CONTEXT_FLAG` means the retrieved label contains no direct disease-context text match and is only a documented screen result. |
| `edges.parquet` | `source_id, relation, target_id` | Existing ReMedy graph: reconciles known indications, off-label uses and contraindications before novelty is claimed. |

## Eligibility logic

1. A pair with an existing indication or off-label-use relation is classified
   as `KNOWN_USE_RECOVERED`; it is useful for retrospective validation but
   excluded from the novel-candidate shortlist.
2. A pair with a contraindication relation or a `FLAGGED` safety context is
   classified as blocked.
3. For a shared drug-target-disease protein, the drug action must oppose the
   disease effect. In compact form, `drug_action_sign × disease_effect_sign = -1`.
4. Missing direction or safety evidence receives an explicit
   `EVIDENCE_INCOMPLETE` or `SAFETY_CONTEXT_UNRESOLVED` status, rather
   than being silently promoted.
5. Only `EVIDENCE_SUPPORTED` pairs are written to
   `evidence_supported_candidates.csv`.

## Colab commands

After pulling the `codex/hybrid-rgcn-training` branch and completing Hybrid
R-GCN training, the full workflow can be run with:

```python
!python src/run_evidence_gated_pipeline.py
```

The following commands are the equivalent transparent step-by-step version:

```python
# 1. Produce source-backed inhibitor/agonist/etc. records for graph drugs.
!python src/preprocessing/fetch_chembl_mechanisms.py

# 2. Run the reproducible label screen. It creates a cached source snapshot.
!python src/preprocessing/fetch_openfda_context_screen.py

# 3. Collect signed disease-target evidence and a mapping audit from Open Targets.
#    Unresolved IDs are withheld rather than guessed.
!python src/preprocessing/fetch_open_targets_direction_evidence.py

# 4. Gate the internal model scores. This is the only command that creates the
#    researcher-facing candidate shortlist.
!python src/validation/evidence_gate.py \
  --ranking reports/model_evaluation/hybrid_rgcn_raw_internal_scores.csv \
  --nodes data/processed/nodes.parquet \
  --edges data/processed/edges.parquet \
  --mechanisms data/processed/drug_mechanisms.csv \
  --disease-effects data/processed/disease_target_effects.csv \
  --safety data/processed/drug_safety_context.csv
```

## Outputs

- `candidate_gate_audit.csv`: every scored pair, its gate status, reason,
  compatible proteins, mechanism evidence IDs and safety evidence ID.
- `evidence_supported_candidates.csv`: the only candidate file intended for
  the product's default researcher view.
- `candidate_gate_summary.json`: pair counts by decision status for reporting.

## Research evaluation

Evaluate the gate independently of the model score:

- mask known drug indications before scoring;
- measure how often they are recovered as `KNOWN_USE_RECOVERED`;
- report the proportion of raw model signals filtered for known use, missing
  evidence, direction conflict and safety flags;
- report evidence coverage and explanation completeness for the final shortlist;
- audit a small, pre-registered top-K sample using cited source records.

This provides a measurable research contribution: ReMedy is assessed not only on
link-prediction accuracy, but on whether the eligibility layer reduces
unsupported hypotheses while keeping source-traceable biological rationale.
