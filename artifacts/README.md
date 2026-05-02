# Artifacts

Runtime outputs should be written here during demo, training, and evaluation runs.

Recommended layout:

```text
artifacts/
  generated/       Generated `.npz` files from Step 1
  mdopt/           UMA/MD optimization and ORCA DFT outputs
  batch_runs/      Batch evaluation outputs
  results/         Training and evaluation result tables
```

These outputs are ignored by git. Keep only small, intentional examples if needed for a release.
