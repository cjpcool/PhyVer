# Training Code

The source release preserves the original training scripts. They are not required for serving the ACL web demo, but are included for reproducibility and inspection.

## Autoencoder

```bash
python train_ae.py
```

The script uses:

```text
datasets/
modules/
utils/
visualization/
```

By default it writes checkpoints under `checkpoints/`. Large checkpoint files are ignored by git.

## Predictor

```bash
python train_predictor.py
```

This script loads the autoencoder checkpoint and trains the prediction head. Confirm that `load_name` and `save_name` inside the script or command-line configuration match your checkpoint layout.

## Agent Generation Script

```bash
python run_modal_agent.py \
  --save_name_ae checkpoints/vae_cond_128_beta001_dis_same_100_frac \
  --save_dir artifacts/results/lattices
```

## Data And Checkpoints

The training scripts currently follow the original research code conventions and may contain local default paths. Before reproducing training:

1. Download or prepare the relevant lattice/materials dataset.
2. Place checkpoints under `checkpoints/`.
3. Redirect generated outputs to `artifacts/`.
4. Keep W&B output under `wandb/`, which is ignored by git.

## Recommended Release Practice

Do not commit:

```text
checkpoints/*.pt
checkpoints/*.pth
checkpoints/*.ckpt
wandb/
artifacts/
```

Commit only scripts, configuration, and small metadata needed to reproduce training.
