# PhyVer

**Authors:** Jianpeng Chen and collaborators
**Institution:** Zhou Lab, Virginia Tech
**ACL Demo 2026**

PhyVer is a physics-grounded verification system for natural-language materials claims. Given a claim, PhyVer generates a candidate material structure, optimizes it with a machine-learned force field, optionally runs DFT characterization, and checks whether the computed evidence supports the claim.

The system pipeline is:

```text
claim -> structure generation -> UMA/MD optimization -> optional ORCA DFT -> claim verification
```

The repository contains the web demo, backend API, generation/optimization/verification code, and the training and evaluation scripts used by the project.

## 1. Using The Demo

Start the server:

```bash
conda activate mat_env
uvicorn server:app --host 0.0.0.0 --port 5557
```

Open:

```text
http://localhost:5557/demo
```

The web demo supports four steps:

1. Generate an initial material structure from a claim.
2. Optimize the structure with UMA/MD.
3. Run optional ORCA DFT characterization.
4. Verify the claim against the generated structure and DFT results.

For a quick smoke test without ORCA, choose:

```text
Mode: rocksalt
No Generator: true
Preset: sprint
Run DFT: false
```

## 2. Environment Preparation

Create and activate the conda environment:

```bash
conda create -n mat_env python=3.11
conda activate mat_env
```

Install core packages:

```bash
pip install numpy scipy pandas scikit-learn tqdm matplotlib plotly
pip install fastapi uvicorn pydantic
pip install ase
pip install openai google-genai
```

Install PyTorch and PyTorch Geometric following the CUDA version on your machine. For example:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

Install Fairchem/UMA dependencies according to the Fairchem instructions:

```bash
pip install fairchem-core
```

ORCA is optional but required for the DFT step. Install ORCA separately and set:

```bash
export ORCA_COMMAND=/path/to/orca
```

## 3. Checkpoints And Runtime Configuration

Expected checkpoint layout:

```text
checkpoints/
  omat24_rattle2/
    best_ae_model.pt
    best_predictor_model.pt
    ...
  uma-s-1p1.pt
```

Download the R_MetaSymbO OMAT24 generation checkpoint from:

```text
https://drive.google.com/drive/folders/1JQ6-tAcz7B5CCfuJSiCyuYng-0eFO9GY?usp=sharing
```

Download the UMA checkpoint from:

```text
https://huggingface.co/facebook/UMA
```

The demo uses these environment variables:

```bash
export DEMO_CKPT_DIR=./checkpoints/omat24_rattle2
export DEMO_SAVE_DIR=./artifacts/generated
export DEMO_OUTDIR=./artifacts/mdopt
export FAIRCHEM_UMA_CKPT=./checkpoints/uma-s-1p1.pt
export FAIRCHEM_UMA_CONFIG=./uma_config.yml
export ORCA_COMMAND=/path/to/orca
```

For LLM-based generation or verification, provide an OpenAI or Gemini API key in the web UI.

## 4. Running Demo, Training, And Evaluation

Run the web demo:

```bash
uvicorn server:app --host 0.0.0.0 --port 5557
```

Run generation from the command line:

```bash
python gen_test.py \
  --no-generator \
  --designer-client gpt-5 \
  --api-key "$OPENAI_API_KEY" \
  --ckpt-dir ./checkpoints/omat24_rattle2 \
  --prompt "Al20Zn80 at 870K is a solid at equilibrium" \
  --save-dir ./artifacts/generated
```

Run UMA optimization:

```bash
python wrap_md_uma.py \
  --gen-path ./artifacts/generated/gen_union.npz \
  --ckpt ./checkpoints/uma-s-1p1.pt \
  --preset sprint \
  --outdir ./artifacts/mdopt
```

Run UMA + ORCA DFT:

```bash
python wrap_md_uma.py \
  --gen-path ./artifacts/generated/gen_union.npz \
  --ckpt ./checkpoints/uma-s-1p1.pt \
  --run-dft \
  --orca-command "$ORCA_COMMAND" \
  --nprocs 8 \
  --maxcore 4000 \
  --outdir ./artifacts/mdopt
```

Run training:

```bash
python train_ae.py
python train_predictor.py
```

Run evaluation:

```bash
python run_phyver_batch_eval.py \
  --gold-jsonl /path/to/gold.jsonl \
  --outdir ./artifacts/batch_runs/phyver_eval

python run_llm_likert_eval.py \
  --gold-jsonl /path/to/gold.jsonl \
  --outdir ./artifacts/batch_runs/llm_eval

python analyze_batch_runs.py --batch-root ./artifacts/batch_runs
```

Runtime outputs are written under `artifacts/` and are ignored by git.

## 5. Citation And License

If you use PhyVer, please cite the ACL Demo 2026 paper associated with this repository.

```bibtex
@inproceedings{phyver2026,
  title = {PhyVer: Physics-Grounded Verification of Natural-Language Materials Claims},
  author = {Chen, Jianpeng and collaborators},
  booktitle = {Proceedings of ACL Demo},
  year = {2026}
}
```

This repository is released with the license in `LICENSE`. Third-party tools, checkpoints, APIs, and datasets are governed by their own licenses and terms.
