# Checkpoints

Large checkpoints are not committed to the source release. Download them separately and place them in this directory.

Expected layout:

```text
checkpoints/
  omat24_rattle2/
    best_ae_model.pt
    best_predictor_model.pt
    ...
  uma-s-1p1.pt
```

## Generation Checkpoint

Download the R_MetaSymbO OMAT24 checkpoint from:

```text
https://drive.google.com/drive/folders/1JQ6-tAcz7B5CCfuJSiCyuYng-0eFO9GY?usp=sharing
```

Place the downloaded files under:

```text
./checkpoints/omat24_rattle2
```

The demo reads this path from:

```bash
export DEMO_CKPT_DIR=./checkpoints/omat24_rattle2
```

## UMA Checkpoint

Download `uma-s-1p1.pt` from the official UMA model repository:

```text
https://huggingface.co/facebook/UMA
```

The model card lists `uma-s-1p1.pt` as the checkpoint for `uma-s-1.1` with MD5 checksum:

```text
36a2f071350be0ee4c15e7ebdd16dde1
```

You can download it with the Hugging Face CLI after requesting access to the gated model:

```bash
huggingface-cli login
huggingface-cli download facebook/UMA checkpoints/uma-s-1p1.pt \
  --local-dir ./checkpoints \
  --local-dir-use-symlinks False
```

After download, place or keep the checkpoint at:

```text
./checkpoints/uma-s-1p1.pt
```

Then set:

```bash
export FAIRCHEM_UMA_CKPT=./checkpoints/uma-s-1p1.pt
```

## Optional UMA Config

If using a Fairchem config file:

```bash
export FAIRCHEM_UMA_CONFIG=./uma_config.yml
```

## Do Not Commit

Do not commit `.pt`, `.pth`, `.ckpt`, `.zip`, `.tar`, or local downloaded model files.
