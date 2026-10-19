# ReMedy evidence-to-disposition workflow

## Purpose

ReMedy uses a graph model to retrieve possible drug-disease hypotheses. The model
score is an internal ranking signal only. A pair is never displayed as a usable
treatment simply because it has a graph path.

The researcher-facing output is a **research disposition**:

- `KNOWN_USE_CONTROL_NOT_NOVEL` — a source-backed indication match; retain only as a control.
- `EXCLUDED_SOURCE_CITED_NEGATIVE_HUMAN_EVIDENCE` — a cited human result was negative.
- `EXCLUDED_SOURCE_CITED_SAFETY_EVIDENCE` — a cited disease-relevant safety reason excludes it.
- `WITHHELD_DISEASE_RELEVANT_SAFETY_REVIEW` — a label-context signal needs review.
- `HUMAN_OUTCOME_ADJUDICATION_REQUIRED` — a human study exists but benefit/harm has not been interpreted.
- `ADVANCE_TO_INDEPENDENT_PRECLINICAL_VALIDATION` — only possible after source-cited causal-target, exposure, and positive phenotype evidence.
- `INSUFFICIENT_EVIDENCE_FOR_RESEARCH_SHORTLIST` — no promotion.

## Scientific guardrail

The Open Targets direction used by the initial gate is a **genetic-risk direction**.
It does not prove that pharmacologically increasing or decreasing the target will
benefit a patient. The curation workspace therefore requires a source-cited review
of causal target role, tissue/exposure, phenotype, human outcome, and safety.

## Kaggle run order

From the repository root:

```bash
%cd /kaggle/working/ReMedy
!git pull --ff-only origin codex/hybrid-rgcn-training
```

If the model and raw ranking already exist, do **not** retrain just to create the
evidence outputs. Run the following in order:

```bash
!python src/preprocessing/fetch_chembl_mechanisms.py
!python src/preprocessing/fetch_open_targets_direction_evidence.py
!python src/modeling/generate_evidence_aware_candidate_rankings.py
!python src/preprocessing/fetch_openfda_context_screen.py \
  --ranking reports/model_evaluation/evidence_aware_rgcn_raw_internal_scores.csv
!python src/validation/evidence_gate.py \
  --ranking reports/model_evaluation/evidence_aware_rgcn_raw_internal_scores.csv \
  --nodes data/processed/nodes.parquet \
  --edges data/processed/edges.parquet \
  --mechanisms data/processed/drug_mechanisms.csv \
  --disease-effects data/processed/disease_target_effects.csv \
  --safety data/processed/drug_safety_context.csv
!python src/curation/build_evidence_chain_workspace.py
```

The last command creates the bounded review workspace. Retrieve context only for
that bounded set:

```bash
!python src/preprocessing/fetch_chembl_indications.py
!python src/preprocessing/fetch_clinical_trial_evidence.py \
  --pairs reports/candidate_eligibility/evidence_supported_candidates.csv
!python src/preprocessing/fetch_pubmed_evidence.py \
  --pairs reports/candidate_eligibility/evidence_supported_candidates.csv
!python src/curation/prefill_evidence_chain_retrievals.py
!python src/curation/build_trial_outcome_packets.py
!python src/curation/evaluate_evidence_chain.py
!python src/curation/build_research_dispositions.py
```

## Outputs

- `reports/evidence_chain/trial_outcome_review_packets.csv` — source-linked
  trial endpoints and review questions. It does not infer efficacy.
- `data/curation/evidence_chain_v1/*.csv` — source-citable curation tables.
- `reports/evidence_chain/candidate_evidence_chain_status.csv` — evidence-gate
  status after citations are entered.
- `reports/researcher_output/candidate_research_dispositions.csv` — final
  researcher-facing stage and next action.

## What requires expert review

A reviewer must enter direct citations before the evaluator will promote or
exclude a candidate. In particular, do not classify a registered trial or a
PubMed retrieval as positive/negative without examining the endpoint, population,
comparator, effect direction, adverse events, and source.
