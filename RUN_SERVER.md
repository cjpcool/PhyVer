# Tutorial for access to fastapi servers
After I started the server in the open port, it is easy to access and run the model via curl.
## Output Json format example:
~~~
{"gen_path":"artifacts/generated/gen_union.npz","optimize":{"uma":{"energy_eV":-3.819211809084268,"traj":"./artifacts/mdopt/best.traj","xyz":"./artifacts/mdopt/best.xyz"},"dft":{"dipole_vec_D":[1.0695197423693514e-05,2.45837361596557e-05,-3.267579437249424e-05],"dipole_D":4.226647453210251e-05,"energy_hartree":-2020.995943482014,"energy_eV":-54994.101219664015,"homo_eV":-3.0082,"lumo_eV":-2.6833,"gap_eV":0.32489999999999997,"mulliken_block":"   0 Al:    0.000000    1.000000\n   1 Zn:    0.000000   -0.000000\nSum of atomic charges         :    0.0000000\nSum of atomic spin populations:    1.0000000","forces":[[-4.666552588461119e-07,-1.077446571746841e-06,1.4285050237735525e-06],[4.4351532865539554e-07,1.0286984521450653e-06,-1.3677755625396948e-06]]},"summary_path":null}}
~~~
In this output, you may need to query keys "uma" and "dft".


## Pipeline Running
~~~
curl -X POST http://zhoulab-1.cs.vt.edu:5557/pipeline -H "Content-Type: application/json" -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./artifacts/generated",
    "no_generator": true,
    "api_key": "openai api key",
    "no-generator": true,

    "gen_path": "./artifacts/generated/gen_union.npz",
    "outdir": "./artifacts/mdopt",
    "run_dft": true,
    "uma_ckpt": "./checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": "sprint" / "quick" / "standard",
    "orca_command": "/path/to/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~


## Step by Step Running
Command for Step 1:

~~~
curl -X POST http://localhost:5557/generate \
  -H "Content-Type: application/json" \
  -d '{
    "claims": "Al20Zn80 at 870K is a solid at equilibrium",
    "ckpt_dir": "./checkpoints/omat24_rattle2",
    "use_llm": true,
    "save_dir": "./artifacts/generated",
    "no_generator": true,
    "api_key": "openai api key",
    "uma"
  }'
  ~~~

Command for Step 2 and 3 (uma optimize + dft computation):
~~~
curl -X POST http://localhost:5557/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "gen_path": "./artifacts/generated/gen_union.npz",
    "outdir": "./artifacts/mdopt",
    "run_dft": true,
    "ckpt": "./checkpoints/uma-s-1p1.pt",
    "loops": 2,
    "preset": ["quick","sprint", "standard", "thorough"],
    "orca_command": "/path/to/orca",
    "orca_maxcore": 4000,
    "orca_nprocs": 8
  }'
~~~


## Website Demo (Claim Verification)

The server now also hosts a browser demo for claim verification and visualization.

### Start server

~~~bash
uvicorn server:app --host 0.0.0.0 --port 5557 --reload
~~~

Open in browser:

~~~
http://localhost:5557/demo
~~~

### Simplified UI inputs

The web UI now only exposes necessary user inputs:

- `claims`
- `api-key`
- `model` (`gpt-5.1` / `gpt-5.1-mini` / `gpt-4.1-mini` / `o4-mini` / `gemini-2.0-flash`)
- `mode` (`llm` / `rocksalt` / `diamond`)
- runtime controls: `no_generator` (default `true`), `run_dft` (default `true`), `loops`, `orca_nprocs` (default `8`), `orca_maxcore` (default `4000`), `device`, `preset`

Checkpoint paths and output paths are hidden and use server defaults.

### Step-by-step running (can rerun from intermediate result)

In the website, use:

1. `Step 1: Generate`
2. `Step 2A: UMA Optimization`
3. `Step 2B: DFT Computation`
3. `Step 3: Verify Claim`
4. `Stop` to cancel all currently running step jobs

At each step, you can inspect visualization/metrics and rerun the same step before proceeding.

During `Step 2A/2B`, print outputs are streamed to the web page under **Optimization Progress Log** for real-time progress tracking.

`Reset` buttons on generated/optimized views now re-read the corresponding structure file and re-visualize.

Each left-side step button routes to the corresponding right-side visualization panel:

- `Step 1: Generate` → Generated Material
- `Step 2A: UMA Optimization` → MetaSymbO Optimized
- `Step 2B: DFT Computation` → DFT Analysis
- `Step 3: Verify Claim` → Claim Verification

### Demo capabilities

1. Visualize generated material from claim
2. Visualize MetaSymbO/UMA optimized material
3. Visualize MD optimization process (energy / volume / RMSD)
4. Show DFT analysis outputs (energy, dipole, HOMO/LUMO/gap, forces)
5. Verify claims by comparing extracted claim constraints against DFT results
6. Select different modes/models/presets/devices in UI

### Demo API endpoints

- `POST /demo/run` : submit asynchronous claim pipeline job
- `GET /demo/defaults` : fetch demo defaults (model/mode/preset/device)
- `POST /demo/step/generate` : run generation only
- `POST /demo/step/optimize/start` : run UMA optimization job (Step 2A)
- `POST /demo/step/dft/start` : run DFT computation job from UMA result (Step 2B)
- `GET /demo/step/jobs/{job_id}` : poll step job status/log/result
- `POST /demo/step/jobs/{job_id}/stop` : stop current step job
- `POST /demo/step/jobs/stop-all` : cancel all running step jobs
- `POST /demo/step/verify` : run claim-vs-DFT verification only
- `GET /demo/jobs/{job_id}` : poll job status
- `GET /demo/jobs/{job_id}/bundle` : fetch all demo artifacts in one response
- `GET /demo/structure?path=...` : parse structure file (`.npz`, `.traj`, `.xyz`)
- `GET /demo/md-metrics?outdir=...` : read/derive optimization metrics
- `GET /demo/dft?summary_path=...` : load DFT/UMA summary fields
- `POST /demo/verify` : standalone claim-vs-DFT verification
