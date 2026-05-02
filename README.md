# PhyVer

PhyVer is a physics-grounded verification system for natural-language materials claims. It converts a user claim into a candidate material structure, optimizes the structure with a machine-learned force field, optionally characterizes it with density functional theory (DFT), and then verifies whether the computed evidence supports the original claim.

The camera-ready paper is included as `ACL_Demo_2026___PhyVer.pdf`.

## Overview

PhyVer is designed for interactive scientific claim checking. Given a claim such as:

```text
Al20Zn80 at 870K is a solid at equilibrium and a semiconductor with a band gap around 0.3 eV.
```

the system runs the following pipeline:

1. **Claim-to-structure generation**
   - Extracts material entities, conditions, and requested properties from the claim.
   - Produces an initial crystal/material structure using the R_MetaSymbO generation stack or an offline prototype scaffold.

2. **Structure relaxation**
   - Converts the generated structure to an ASE `Atoms` object.
   - Runs an anneal, quench, and 0 K relaxation loop with a Fairchem/UMA calculator.
   - Falls back to ASE EMT when configured and UMA is unavailable.

3. **DFT characterization**
   - Runs optional ORCA single-point DFT on the optimized structure.
   - Extracts energy, dipole, HOMO/LUMO values, band gap, and forces.

4. **Claim verification**
   - Compares the original claim against the generated structure and DFT outputs.
   - Produces a verdict, Likert score, extracted constraints, and parameter-level checks.

5. **Web demonstration**
   - Provides a browser interface for step-by-step execution, visualization, log streaming, DFT inspection, and verification review.

## Repository Organization

```text
web_demo/                     Browser UI for the ACL demo
server.py                     FastAPI backend used by the browser demo
gen_test.py                   Claim-to-structure generation entrypoint
wrap_md_uma.py                UMA/MD optimization and ORCA DFT wrapper
claim_verification_llm.py     Claim-vs-DFT verification logic

model/                        LLM/material generation agents
modules/                      Geometric autoencoder, diffusion, backbone modules
datasets/                     Dataset wrappers and lattice data structures
utils/                        Materials, lattice, and LLM helpers
visualization/                Structure and lattice visualization utilities
structure_optim_modules/      MD optimization and trajectory analysis code
evaluation/                   Generation and prediction evaluation utilities
baselines/                    Baseline prompting/guidance code

train_ae.py                   Autoencoder training script
train_predictor.py            Predictor training script
run_modal_agent.py            Agent-based generation script
batch_claim_pipeline.py       Batch claim -> generation -> optimization pipeline
analyze_batch_runs.py         Batch run analysis
run_llm_likert_eval.py        LLM-based Likert evaluation
run_phyver_batch_eval.py      PhyVer batch evaluation

forcefields.yml               Conda environment specification for `mat_env`
uma_config.yml                Optional UMA/Fairchem config
checkpoints/README.md         Checkpoint download and layout instructions
artifacts/README.md           Runtime output layout
```

Generated outputs, local logs, W&B runs, virtual environments, and large model binaries are intentionally ignored by git. See `.gitignore`.

## Implementation Details

The implementation keeps the demo entrypoints at the repository root so the original research scripts remain directly runnable.

### Web Demo

The frontend lives in `web_demo/`:

```text
web_demo/index.html
web_demo/app.js
web_demo/styles.css
web_demo/logo.png
```

It uses 3Dmol.js for structure visualization and Plotly for optimization metrics. The frontend calls FastAPI endpoints in `server.py`.

### Backend API

`server.py` defines the demo API and job orchestration:

```text
GET  /demo
POST /demo/step/generate
POST /demo/step/optimize/start
POST /demo/step/dft/start
POST /demo/step/verify
GET  /demo/step/jobs/{job_id}
POST /demo/step/jobs/stop-all
GET  /demo/structure
GET  /demo/md-metrics
GET  /demo/dft
```

Optimization and DFT jobs are launched asynchronously and polled by the frontend so long-running computations can stream progress logs.

### Generation

`gen_test.py` is the generation entrypoint. It supports:

```text
llm       LLM-based claim/entity/scaffold generation
rocksalt  Offline prototype scaffold
diamond   Offline prototype scaffold
```

The main model and agent code is in:

```text
model/
modules/
datasets/
utils/
visualization/
```

The generated structure is saved as `.npz` with:

```text
atom_types
cart_coords
lengths
angles
```

### Optimization And DFT

`wrap_md_uma.py` reads generated `.npz` structures, constructs ASE atoms, attaches a UMA/Fairchem calculator when available, and runs MD relaxation through `structure_optim_modules/`.

The same wrapper can optionally run ORCA DFT and write:

```text
best.traj
best.xyz
best_energy.txt
orca_sp/dft_results.json
summary.json
```

### Verification

`claim_verification_llm.py` compares claim constraints against computed evidence. It can call OpenAI or Gemini models and includes a heuristic fallback for common properties such as band gap, metallic/semiconductor/insulator claims, and dipole magnitude.

### Training And Evaluation

Training scripts are preserved:

```text
train_ae.py
train_predictor.py
run_modal_agent.py
```

Evaluation scripts are preserved:

```text
batch_claim_pipeline.py
analyze_batch_runs.py
run_llm_likert_eval.py
run_phyver_batch_eval.py
evaluation/
baselines/
```

Detailed instructions are in:

```text
docs/RUN_DEMO.md
docs/RUN_TRAINING.md
docs/RUN_EVALUATION.md
```

## Reimplementation Guide

To reimplement PhyVer from this source release, reproduce the four system layers below.

1. **Set up the model/runtime environment**
   - Install the `mat_env` conda environment from `forcefields.yml`.
   - Install ORCA separately if DFT is required.
   - Download the generation and UMA checkpoints described in `checkpoints/README.md`.

2. **Reproduce generation**
   - Use `gen_test.py` as the reference implementation.
   - Start with offline `rocksalt` or `diamond` mode to validate structure serialization.
   - Enable LLM mode after configuring an OpenAI or Gemini API key.

3. **Reproduce physical optimization**
   - Use `wrap_md_uma.py` as the reference implementation.
   - Confirm that generated `.npz` structures can be converted to ASE atoms.
   - Run UMA/MD relaxation and verify that `best.traj`, `best.xyz`, and `summary.json` are produced.

4. **Reproduce verification**
   - Use `claim_verification_llm.py` to convert DFT outputs into claim-level verdicts.
   - Compare the returned `checks`, `extracted_constraints`, and Likert `score` against the claim.

5. **Reproduce the web demo**
   - Serve `server.py` with Uvicorn.
   - Open `/demo` and execute the four UI steps.
   - Inspect generated structures, optimized structures, DFT cards, and verification tables.

For detailed command-level instructions, use the task-specific docs:

```text
docs/RUN_DEMO.md
docs/RUN_TRAINING.md
docs/RUN_EVALUATION.md
```

## Installation

Create the environment from the provided conda file:

```bash
conda env create -f forcefields.yml
conda activate mat_env
```

If you manage dependencies manually, the core runtime packages are Python 3.9+, PyTorch, PyTorch Geometric, ASE, FastAPI, Uvicorn, NumPy, OpenAI or Google GenAI client libraries, and Fairchem/OCP for UMA.

ORCA is optional but required for the DFT step. Install ORCA separately and expose its binary through `ORCA_COMMAND`.

## Checkpoints

Create this layout:

```text
checkpoints/
  omat24_rattle2/
    best_ae_model.pt
    best_predictor_model.pt
    ...
  uma-s-1p1.pt
```

Generation checkpoint:

```text
Download the R_MetaSymbO OMAT24 checkpoint from:
https://drive.google.com/drive/folders/1JQ6-tAcz7B5CCfuJSiCyuYng-0eFO9GY?usp=sharing

Place it under:
./checkpoints/omat24_rattle2
```

UMA checkpoint:

```text
Download `uma-s-1p1.pt` from the official UMA model repository and place it at:

```text
https://huggingface.co/facebook/UMA
```

Expected local path:

```text
./checkpoints/uma-s-1p1.pt
```

More details are in `checkpoints/README.md`.

## Environment Variables

Set these before running the full demo:

```bash
export DEMO_CKPT_DIR=./checkpoints/omat24_rattle2
export DEMO_SAVE_DIR=./artifacts/generated
export DEMO_OUTDIR=./artifacts/mdopt
export FAIRCHEM_UMA_CKPT=./checkpoints/uma-s-1p1.pt
export FAIRCHEM_UMA_CONFIG=./uma_config.yml
export ORCA_COMMAND=/path/to/orca
```

For LLM-backed generation or verification, enter the API key in the web UI or provide it through your own deployment secret manager. Do not commit API keys.

## Running The ACL Demo

Start the FastAPI server:

```bash
uvicorn server:app --host 0.0.0.0 --port 5557
```

Open:

```text
http://localhost:5557/demo
```

The browser demo implements:

1. Claim-driven material generation.
2. UMA/MLFF optimization with streamed progress logs.
3. ORCA DFT characterization.
4. Claim verification against generated structure and DFT outputs.
5. Structure visualization, MD metric plots, refresh/reset controls, and stop controls.

The UI supports three generation modes:

```text
llm       Uses an OpenAI/Gemini model and API key.
rocksalt  Offline prototype scaffold.
diamond   Offline prototype scaffold.
```

For a quick server/UI smoke test without ORCA, use `rocksalt` or `diamond`, set `Run DFT=false`, and use the `sprint` preset.

See `docs/RUN_DEMO.md` for endpoint details and troubleshooting.

## Training

Autoencoder training:

```bash
python train_ae.py
```

Predictor training:

```bash
python train_predictor.py
```

Agent generation:

```bash
python run_modal_agent.py \
  --save_name_ae checkpoints/vae_cond_128_beta001_dis_same_100_frac \
  --save_dir artifacts/results/lattices
```

The training scripts expect the dataset/checkpoint paths used by the original project. Before publishing or reproducing experiments, update dataset paths in the script arguments or environment to match your local machine.

See `docs/RUN_TRAINING.md`.

## Evaluation

Batch PhyVer evaluation:

```bash
python run_phyver_batch_eval.py \
  --gold-jsonl ./sprint1-drop4-gold-standards.jsonl \
  --outdir ./artifacts/batch_runs/phyver_eval_smoke
```

LLM Likert evaluation:

```bash
python run_llm_likert_eval.py \
  --gold-jsonl ./sprint1-drop4-gold-standards.jsonl \
  --outdir ./artifacts/batch_runs/llm_eval_smoke
```

Full claim pipeline:

```bash
python batch_claim_pipeline.py \
  --jsonl ./sprint1-drop4-problems.jsonl \
  --ckpt-gen ./checkpoints/omat24_rattle2 \
  --uma-ckpt ./checkpoints/uma-s-1p1.pt \
  --output-root ./artifacts/batch_runs \
  --preset quick \
  --limit 5
```

See `docs/RUN_EVALUATION.md`.

## Runtime Outputs

Runtime outputs should be written under:

```text
artifacts/generated/      Generated `.npz` structures
artifacts/mdopt/          UMA, trajectory, and DFT outputs
artifacts/batch_runs/     Batch evaluation outputs
artifacts/results/        Training/evaluation result tables
```

Legacy defaults such as `_gens/`, `_mdopt/`, `batch_runs/`, `results/`, and `wandb/` are still ignored so older commands do not pollute the release.

## Artifact Policy

Do not commit checkpoints, ORCA binaries, API keys, W&B runs, generated trajectories, DFT outputs, or local logs. The `.gitignore` file keeps these outputs out of the source release while preserving documentation and placeholder directories.

`wrap_md_uma.py` is the active optimization and DFT wrapper used by the demo and evaluation scripts.
