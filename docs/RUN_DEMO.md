# Running The PhyVer Web Demo

The ACL demo is served by `server.py` and the static files in `web_demo/`.

## 1. Prepare Checkpoints

Expected layout:

```text
checkpoints/
  omat24_rattle2/
    best_ae_model.pt
    best_predictor_model.pt
    ...
  uma-s-1p1.pt
```

See `checkpoints/README.md` for download notes.

## 2. Configure Runtime Paths

```bash
export DEMO_CKPT_DIR=./checkpoints/omat24_rattle2
export DEMO_SAVE_DIR=./artifacts/generated
export DEMO_OUTDIR=./artifacts/mdopt
export FAIRCHEM_UMA_CKPT=./checkpoints/uma-s-1p1.pt
export FAIRCHEM_UMA_CONFIG=./uma_config.yml
export ORCA_COMMAND=/path/to/orca
```

`ORCA_COMMAND` is only needed when running the DFT step.

## 3. Start Server

```bash
uvicorn server:app --host 0.0.0.0 --port 5557
```

Open:

```text
http://localhost:5557/demo
```

## 4. Demo Workflow

1. `Step 1: Generate`
   - Calls `POST /demo/step/generate`.
   - Produces a generated `.npz` structure.
   - Visualizes the generated structure with 3Dmol.

2. `Step 2: UMA Optimization`
   - Calls `POST /demo/step/optimize/start`.
   - Runs `wrap_md_uma.py` asynchronously.
   - Polls `GET /demo/step/jobs/{job_id}`.
   - Visualizes optimized structure and plots energy, volume, and RMSD.

3. `Step 3: DFT Computation`
   - Calls `POST /demo/step/dft/start`.
   - Runs ORCA single-point DFT from `best.traj`.
   - Displays energy, band gap, HOMO/LUMO, dipole, and forces.

4. `Step 4: Verify Claim`
   - Calls `POST /demo/step/verify`.
   - Uses `claim_verification_llm.py`.
   - Displays verdict, Likert score, constraints, and parameter checks.

## 5. Smoke Test Without ORCA

Use these UI settings:

```text
Mode: rocksalt
No Generator: true
Preset: sprint
Run DFT: false
Loops: 1
```

Then run Step 1 and Step 2. This checks that the server, static demo, generation path, and UMA/EMT fallback path are wired correctly.

## 6. Main Demo Endpoints

```text
GET  /demo
GET  /demo/defaults
POST /demo/step/generate
POST /demo/step/optimize/start
POST /demo/step/dft/start
GET  /demo/step/jobs/{job_id}
POST /demo/step/jobs/{job_id}/stop
POST /demo/step/jobs/stop-all
POST /demo/step/verify
GET  /demo/structure?path=...
GET  /demo/md-metrics?outdir=...
GET  /demo/dft?summary_path=...
```

## 7. Troubleshooting

- If `wrap_md_uma import failed`, confirm `ase`, `torch`, and Fairchem dependencies are installed.
- If DFT fails, confirm `ORCA_COMMAND` points to an executable ORCA binary.
- If LLM generation or verification fails, confirm the API key and selected model are valid.
- If no structure appears, check `DEMO_SAVE_DIR`, `DEMO_OUTDIR`, and browser console errors.
