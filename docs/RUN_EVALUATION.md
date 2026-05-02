# Evaluation Code

The source release preserves both batch PhyVer evaluation and LLM Likert evaluation.

## Inputs

The repository includes:

```text
sprint1-drop4-problems.jsonl
sprint1-drop4-gold-standards.jsonl
```

`sprint1-drop4-problems.jsonl` contains claim prompts. `sprint1-drop4-gold-standards.jsonl` contains the gold/evaluation records used by evaluation scripts.

## PhyVer Batch Evaluation

```bash
python run_phyver_batch_eval.py \
  --gold-jsonl ./sprint1-drop4-gold-standards.jsonl \
  --outdir ./artifacts/batch_runs/phyver_eval_smoke
```

This script uses:

```text
run_phyver_batch_eval.py
batch_claim_pipeline.py
claim_verification_llm.py
wrap_md_uma.py
gen_test.py
```

## LLM Likert Evaluation

```bash
python run_llm_likert_eval.py \
  --gold-jsonl ./sprint1-drop4-gold-standards.jsonl \
  --outdir ./artifacts/batch_runs/llm_eval_smoke
```

This script calls the LLM judge helper in `claim_verification_llm.py`.

## Full Claim Pipeline

```bash
python batch_claim_pipeline.py \
  --jsonl ./sprint1-drop4-problems.jsonl \
  --ckpt-gen ./checkpoints/omat24_rattle2 \
  --uma-ckpt ./checkpoints/uma-s-1p1.pt \
  --output-root ./artifacts/batch_runs \
  --preset quick \
  --limit 5
```

Use `--dft` and `--orca-command /path/to/orca` when ORCA is available and DFT outputs are required.

## Analysis

```bash
python analyze_batch_runs.py \
  --batch-root ./artifacts/batch_runs \
  --outdir ./artifacts/batch_runs/analysis_summary
```

## Output Policy

Evaluation outputs should go under:

```text
artifacts/batch_runs/
```

The legacy `batch_runs/` path is still ignored for backward compatibility.
