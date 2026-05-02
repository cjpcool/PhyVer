"""
FastAPI server exposing R_MetaSymbO crystal generation + UMA/DFT optimization.
"""
# pip install fastapi uvicorn pydantic
from __future__ import annotations

import glob
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ase.data import chemical_symbols
    from ase.io import read as ase_read
    HAS_ASE = True
except Exception:
    HAS_ASE = False
    chemical_symbols = None
    ase_read = None

# === Optional: import the programmatic pipeline for UMA + ORCA ===
# Make sure your PYTHONPATH can find wrap_md_uma.py
try:
    from wrap_md_uma import optimize_and_characterize  # noqa: F401
    HAS_OPTIMIZER = True
except Exception as e:
    HAS_OPTIMIZER = False
    _IMPORT_ERR = e

try:
    from claim_verification_llm import verify_claim_with_llm
    HAS_LLM_VERIFIER = True
except Exception:
    HAS_LLM_VERIFIER = False

app = FastAPI(
    title="R_MetaSymbO Crystal Pipeline API",
    description="Expose generation (LLM scaffold/offline) and UMA/DFT optimization via HTTP.",
    version="0.1.0",
)

DEMO_ROOT = Path("./web_demo")
DEMO_JOBS: Dict[str, Dict[str, Any]] = {}
DEMO_LOCK = threading.Lock()
DEMO_STEP_JOBS: Dict[str, Dict[str, Any]] = {}
DEMO_STEP_PROCS: Dict[str, subprocess.Popen] = {}
DEMO_DEFAULT_CKPT_DIR = os.getenv("DEMO_CKPT_DIR", "./checkpoints/omat24_rattle2")
DEMO_DEFAULT_SAVE_DIR = os.getenv("DEMO_SAVE_DIR", "./artifacts/generated")
DEMO_DEFAULT_OUTDIR = os.getenv("DEMO_OUTDIR", "./artifacts/mdopt")
DEMO_DEFAULT_UMA_CKPT = os.getenv("FAIRCHEM_UMA_CKPT", "./checkpoints/uma-s-1p1.pt")
DEMO_DEFAULT_MODEL = os.getenv("DEMO_DESIGNER_MODEL", "gpt-5.1")

# ---------- Multi-user isolation ----------
# Serialize GPU-heavy work so concurrent users don't OOM
GPU_SEMAPHORE = threading.Semaphore(int(os.getenv("DEMO_GPU_CONCURRENCY", "1")))

# TTL for per-session working directories (seconds, default 24 h)
SESSION_TTL = int(os.getenv("DEMO_SESSION_TTL", str(24 * 3600)))


def _session_save_dir(session_id: Optional[str]) -> str:
    """Return per-session generation dir, falling back to the global default."""
    if session_id:
        return str(Path(DEMO_DEFAULT_SAVE_DIR) / session_id)
    return DEMO_DEFAULT_SAVE_DIR


def _session_outdir(session_id: Optional[str]) -> str:
    """Return per-session optimization dir, falling back to the global default."""
    if session_id:
        return str(Path(DEMO_DEFAULT_OUTDIR) / session_id)
    return DEMO_DEFAULT_OUTDIR


def _cleanup_old_sessions():
    """Remove session sub-directories older than SESSION_TTL."""
    for base in [DEMO_DEFAULT_SAVE_DIR, DEMO_DEFAULT_OUTDIR]:
        root = Path(base)
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir():
                try:
                    if (time.time() - d.stat().st_mtime) > SESSION_TTL:
                        shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass


def _start_cleanup_timer():
    """Run _cleanup_old_sessions every hour in a daemon thread."""
    def _loop():
        while True:
            try:
                _cleanup_old_sessions()
            except Exception:
                pass
            time.sleep(3600)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

# Start the periodic cleaner on module load
_start_cleanup_timer()

# ---------- Models ----------
class GenerateIn(BaseModel):
    claims: str  # the prompt/claims for generation
    ckpt_dir: str
    cif_dir: Optional[str] = None
    save_dir: str = "./artifacts/generated"
    use_llm: bool = True  # if False, choose a prototype
    no_generator: bool = False
    prototype: Optional[str] = None  # "rocksalt" / "diamond" when use_llm=False
    api_key: Optional[str] = None  # if None, use env api_key
    extra_args: Optional[List[str]] = None  # passthrough CLI args, e.g. ["--foo","bar"]
    designer_client: Optional[str] = None

class GenerateOut(BaseModel):
    gen_path: str
    saved_files: List[str]
    log: Optional[str] = None

class OptimizeIn(BaseModel):
    gen_path: str
    outdir: str = "./artifacts/mdopt"
    device: Optional[str] = "cuda"
    preset: str = "standard"  # "quick" | "standard" | "thorough"
    loops: int = 5
    run_dft: bool = False
    uma_ckpt: Optional[str] = None          # or env FAIRCHEM_UMA_CKPT
    uma_config_yml: Optional[str] = None    # or env FAIRCHEM_UMA_CONFIG
    orca_command: Optional[str] = None      # or env ORCA_COMMAND
    orca_maxcore: Optional[int] = 4000
    orca_nprocs: Optional[int] = 8
    orca_simpleinput: Optional[str] = None  # or use default

class OptimizeOut(BaseModel):
    uma: Dict[str, Any]
    dft: Optional[Dict[str, Any]] = None
    summary_path: Optional[str] = None

class PipelineIn(BaseModel):
    # generate
    claims: str
    ckpt_dir: str
    cif_dir: Optional[str] = None
    save_dir: str = "./artifacts/generated"
    use_llm: bool = True
    no_generator: bool = False
    prototype: Optional[str] = None
    api_key: Optional[str] = None
    extra_args: Optional[List[str]] = None
    designer_client: Optional[str] = None
    # optimize
    outdir: str = "./artifacts/mdopt"
    device: Optional[str] = "cuda"
    preset: str = "standard"
    loops: int = 5
    run_dft: bool = False
    uma_ckpt: Optional[str] = None
    uma_config_yml: Optional[str] = None
    orca_command: Optional[str] = None
    orca_maxcore: Optional[int] = 4000
    orca_nprocs: Optional[int] = 8
    orca_simpleinput: Optional[str] = None  # or use default


class PipelineOut(BaseModel):
    gen_path: str
    optimize: Optional[OptimizeOut] = None


class DemoRunIn(PipelineIn):
    pass


class DemoVerifyIn(BaseModel):
    claim: str
    dft: Optional[Dict[str, Any]] = None
    summary_path: Optional[str] = None
    gen_path: Optional[str] = None
    designer_client: Optional[str] = DEMO_DEFAULT_MODEL
    api_key: Optional[str] = None


class DemoVerifyOut(BaseModel):
    verdict: str
    score: float
    reason: Optional[str] = None
    checks: List[Dict[str, Any]]
    extracted_constraints: List[Dict[str, Any]]
    parameter_comparisons: Optional[List[Dict[str, Any]]] = None
    dft_used: Dict[str, Any]
    model_used: Optional[str] = None


class DemoStepGenerateIn(BaseModel):
    claims: str
    api_key: Optional[str] = None
    model: Optional[str] = DEMO_DEFAULT_MODEL
    mode: str = "llm"  # llm | rocksalt | diamond
    no_generator: bool = True
    session_id: Optional[str] = None


class DemoStepOptimizeIn(BaseModel):
    gen_path: Optional[str] = None
    preset: str = "standard"
    device: str = "cuda"
    run_dft: bool = True
    loops: int = 5
    orca_nprocs: int = 8
    orca_maxcore: int = 4000
    claim: Optional[str] = None
    summary_path: Optional[str] = None
    designer_client: Optional[str] = DEMO_DEFAULT_MODEL
    api_key: Optional[str] = None
    session_id: Optional[str] = None


class DemoStepDftIn(BaseModel):
    outdir: str = DEMO_DEFAULT_OUTDIR
    orca_nprocs: int = 8
    orca_maxcore: int = 4000
    claim: Optional[str] = None
    gen_path: Optional[str] = None
    designer_client: Optional[str] = DEMO_DEFAULT_MODEL
    api_key: Optional[str] = None
    session_id: Optional[str] = None


class DemoStepVerifyIn(BaseModel):
    claim: str
    summary_path: Optional[str] = None
    gen_path: Optional[str] = None
    designer_client: Optional[str] = DEMO_DEFAULT_MODEL
    api_key: Optional[str] = None
    session_id: Optional[str] = None


class DemoStopAllIn(BaseModel):
    session_id: Optional[str] = None


class DemoStepJobOut(BaseModel):
    job_id: str
    status: str
    log: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# ---------- Helpers ----------
def _ensure_exists(path: str | Path, kind: str = "file/dir"):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"{kind} not found: {path}")
    return p

def _find_newest_npz(save_dir: str | Path) -> Optional[Path]:
    p = Path(save_dir)
    cands = sorted(p.glob("*.npz"), key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def _run_gen_test(
    claims: str,
    ckpt_dir: str,
    cif_dir: Optional[str],
    save_dir: str,
    use_llm: bool,
    prototype: Optional[str],
    no_generator: bool,
    api_key: Optional[str],
    extra_args: Optional[List[str]],
    designer_client: Optional[str] = None,
) -> tuple[Path, List[str], str]:
    """
    Calls the README's generation entrypoint:
      - LLM scaffold:  python gen_test.py --cif-dir ... --ckpt-dir ... --prompt "<claims>" --save-dir ...
      - Offline:       python gen_test.py --no-llm --prototype rocksalt --ckpt-dir ... --save-dir ...
    Returns: (newest_npz, all_saves, log_text)
    """
    env = os.environ.copy()

    cmd: List[str] = [sys.executable, "gen_test.py", "--ckpt-dir", ckpt_dir, "--save-dir", save_dir]
    if use_llm:
        cmd += ["--prompt", claims]
        if api_key:
            cmd += ["--api-key", api_key]
        if cif_dir:
            cmd += ["--cif-dir", cif_dir]
        if designer_client:
            cmd += ["--designer-client", designer_client]
    else:
        # Offline prototype
        if not prototype:
            raise HTTPException(status_code=400, detail="prototype is required when use_llm=False.")
        cmd += ["--no-llm", "--prototype", prototype]

    if no_generator:
        cmd += ["--no-generator"]

    if extra_args:
        cmd += list(extra_args)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Run
    t0 = time.time()
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"gen_test.py failed:\n{e.output}") from e

    # Resolve newest .npz produced
    newest = _find_newest_npz(save_dir)
    if newest is None:
        # Sometimes users output .npy; try a broader message
        raise HTTPException(
            status_code=500,
            detail=f"Generation finished but no .npz found in {save_dir}. Check logs:\n{out}",
        )

    files = [str(p) for p in Path(save_dir).glob("*")]
    log = f"[elapsed: {time.time()-t0:.2f}s]\n{out}"
    return newest, files, log


def _find_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atoms_to_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Structure file not found: {path}")

    if p.suffix.lower() == ".npz":
        npz = np.load(str(p))
        zs = npz["atom_types"].astype(int).tolist()
        coords = npz["cart_coords"].astype(float).tolist()
        lengths = npz["lengths"].astype(float).reshape(-1).tolist() if "lengths" in npz else None
        angles = npz["angles"].astype(float).reshape(-1).tolist() if "angles" in npz else None
        atoms = []
        for i, z in enumerate(zs):
            sym = chemical_symbols[z] if HAS_ASE and chemical_symbols and z < len(chemical_symbols) else str(z)
            x, y, zc = coords[i]
            atoms.append({"Z": int(z), "symbol": sym, "x": float(x), "y": float(y), "z": float(zc)})
        xyz_lines = [str(len(atoms)), f"source={path}"]
        xyz_lines += [f"{a['symbol']} {a['x']:.8f} {a['y']:.8f} {a['z']:.8f}" for a in atoms]
        return {
            "source": path,
            "format": "npz",
            "n_atoms": len(atoms),
            "lengths": lengths,
            "angles": angles,
            "atoms": atoms,
            "xyz_text": "\n".join(xyz_lines) + "\n",
        }

    if not HAS_ASE:
        raise HTTPException(status_code=500, detail="ASE not available for reading non-NPZ structure files.")

    try:
        atoms_obj = ase_read(str(p))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse structure {path}: {e}") from e

    zs = atoms_obj.get_atomic_numbers().tolist()
    coords = atoms_obj.get_positions().tolist()
    atoms = []
    for i, z in enumerate(zs):
        sym = chemical_symbols[z] if chemical_symbols and z < len(chemical_symbols) else str(z)
        x, y, zc = coords[i]
        atoms.append({"Z": int(z), "symbol": sym, "x": float(x), "y": float(y), "z": float(zc)})
    cell_arr = np.array(atoms_obj.cell.array).tolist() if atoms_obj.cell is not None else None
    pbc = [bool(x) for x in getattr(atoms_obj, "pbc", [False, False, False])]
    xyz_lines = [str(len(atoms)), f"source={path}"]
    xyz_lines += [f"{a['symbol']} {a['x']:.8f} {a['y']:.8f} {a['z']:.8f}" for a in atoms]
    return {
        "source": path,
        "format": p.suffix.lower().lstrip("."),
        "n_atoms": len(atoms),
        "cell": cell_arr,
        "pbc": pbc,
        "atoms": atoms,
        "xyz_text": "\n".join(xyz_lines) + "\n",
    }


def _load_md_metrics(outdir: str) -> Dict[str, Any]:
    root = Path(outdir)
    metrics_path = root / "analysis" / "metrics.json"
    data = _find_json(metrics_path)
    if data is not None:
        return {"source": str(metrics_path), "metrics": data}

    if not HAS_ASE:
        return {"source": None, "metrics": {"loop": [], "energy_eV": [], "volume_A3": [], "rmsd_A": []}}

    trajs = sorted(glob.glob(str(root / "loop_*.traj")))
    loops: List[int] = []
    energies: List[float] = []
    volumes: List[float] = []
    positions: List[np.ndarray] = []

    for t in trajs:
        try:
            atoms_obj = ase_read(t)
            loop_num = int(Path(t).stem.split("_")[1])
            loops.append(loop_num)
            try:
                energies.append(float(atoms_obj.get_potential_energy()))
            except Exception:
                energies.append(float("nan"))
            try:
                volumes.append(float(atoms_obj.get_volume()))
            except Exception:
                volumes.append(float("nan"))
            positions.append(np.array(atoms_obj.get_positions(), dtype=float))
        except Exception:
            continue

    rmsd = []
    for i in range(1, len(positions)):
        try:
            rmsd_val = float(np.sqrt(np.mean((positions[i] - positions[i - 1]) ** 2)))
        except Exception:
            rmsd_val = float("nan")
        rmsd.append(rmsd_val)

    return {
        "source": "derived_from_loop_traj",
        "metrics": {
            "loop": loops,
            "energy_eV": energies,
            "volume_A3": volumes,
            "rmsd_A": rmsd,
        },
    }


def _get_structure_context_for_verify(gen_path: Optional[str]) -> Optional[Dict[str, Any]]:
    path = gen_path or _latest_generated_path()
    if not path:
        return None
    try:
        return _atoms_to_json(path)
    except Exception:
        return None


def _verify_claim_against_dft(
    claim: str,
    dft: Dict[str, Any],
    gen_path: Optional[str] = None,
    designer_client: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if not HAS_LLM_VERIFIER:
        return {
            "verdict": "insufficient-evidence",
            "score": 0.0,
            "reason": "LLM verifier module unavailable.",
            "checks": [],
            "extracted_constraints": [],
            "parameter_comparisons": [],
            "dft_used": dft,
            "model_used": None,
        }

    structure_ctx = _get_structure_context_for_verify(gen_path)
    return verify_claim_with_llm(
        claim=claim,
        generated_structure=structure_ctx,
        dft=dft,
        designer_client=designer_client or DEMO_DEFAULT_MODEL,
        api_key=api_key,
    )


def _pipeline_sync(inp: PipelineIn) -> Dict[str, Any]:
    out = pipeline(inp)
    return out.model_dump() if hasattr(out, "model_dump") else out.dict()


def _get_job(job_id: str) -> Dict[str, Any]:
    with DEMO_LOCK:
        job = DEMO_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


def _start_pipeline_job(inp: PipelineIn) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    payload = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    with DEMO_LOCK:
        DEMO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "request": payload,
            "result": None,
            "error": None,
        }

    def _worker():
        with DEMO_LOCK:
            DEMO_JOBS[job_id]["status"] = "running"
            DEMO_JOBS[job_id]["updated_at"] = time.time()
        try:
            res = _pipeline_sync(inp)
            with DEMO_LOCK:
                DEMO_JOBS[job_id]["status"] = "completed"
                DEMO_JOBS[job_id]["result"] = res
                DEMO_JOBS[job_id]["updated_at"] = time.time()
        except Exception as e:
            with DEMO_LOCK:
                DEMO_JOBS[job_id]["status"] = "failed"
                DEMO_JOBS[job_id]["error"] = {
                    "message": str(e),
                    "trace": traceback.format_exc(limit=4),
                }
                DEMO_JOBS[job_id]["updated_at"] = time.time()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}


def _latest_generated_path() -> Optional[str]:
    newest = _find_newest_npz(DEMO_DEFAULT_SAVE_DIR)
    return str(newest) if newest else None


def _step_mode_to_generate_args(mode: str) -> Dict[str, Any]:
    mode_norm = (mode or "llm").strip().lower()
    if mode_norm == "llm":
        return {"use_llm": True, "prototype": None}
    if mode_norm in ("rocksalt", "diamond"):
        return {"use_llm": False, "prototype": mode_norm}
    raise HTTPException(status_code=400, detail="mode must be one of: llm, rocksalt, diamond")


def _create_step_job(kind: str, payload: Dict[str, Any], session_id: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with DEMO_LOCK:
        DEMO_STEP_JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "request": payload,
            "session_id": session_id,
            "log": "",
            "result": None,
            "error": None,
        }
    return job_id


def _append_step_job_log(job_id: str, text: str):
    if not text:
        return
    with DEMO_LOCK:
        job = DEMO_STEP_JOBS.get(job_id)
        if not job:
            return
        job["log"] = (job.get("log", "") + text)[-200000:]
        job["updated_at"] = time.time()


class _StepJobLogWriter(io.TextIOBase):
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id

    def write(self, s: str):
        _append_step_job_log(self.job_id, s)
        return len(s)

    def flush(self):
        return


def _get_step_job(job_id: str) -> Dict[str, Any]:
    with DEMO_LOCK:
        job = DEMO_STEP_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Step job not found: {job_id}")
    return job


def _build_optimize_result_from_job_request(req: Dict[str, Any]) -> Dict[str, Any]:
    gen_path = req.get("gen_path")
    outdir = req.get("outdir", DEMO_DEFAULT_OUTDIR)
    claim = req.get("claim")
    summary_path = str(Path(outdir) / "summary.json")
    summary = _find_json(Path(summary_path))

    optimize_dict: Dict[str, Any] = {
        "summary_path": summary_path if Path(summary_path).is_file() else None,
    }
    if summary:
        optimize_dict["uma"] = summary.get("uma") or {}
        optimize_dict["dft"] = summary.get("dft")

    best_traj = Path(outdir) / "best.traj"
    optimized = _atoms_to_json(str(best_traj)) if best_traj.is_file() else None
    metrics = _load_md_metrics(outdir)

    verification = None
    if claim and summary and summary.get("dft"):
        verification = _verify_claim_against_dft(
            claim,
            summary.get("dft") or {},
            gen_path=gen_path,
            designer_client=req.get("designer_client"),
            api_key=req.get("api_key"),
        )

    return {
        "step": "optimize",
        "gen_path": gen_path,
        "outdir": outdir,
        "optimize": optimize_dict,
        "optimized": optimized,
        "md_metrics": metrics,
        "summary": summary,
        "verification": verification,
    }


def _build_uma_result_from_job_request(req: Dict[str, Any]) -> Dict[str, Any]:
    gen_path = req.get("gen_path")
    outdir = req.get("outdir", DEMO_DEFAULT_OUTDIR)
    summary_path = str(Path(outdir) / "summary.json")
    summary = _find_json(Path(summary_path))

    optimize_dict: Dict[str, Any] = {
        "summary_path": summary_path if Path(summary_path).is_file() else None,
        "uma": (summary or {}).get("uma") or {},
        "dft": None,
    }

    best_traj = Path(outdir) / "best.traj"
    optimized = _atoms_to_json(str(best_traj)) if best_traj.is_file() else None
    metrics = _load_md_metrics(outdir)

    return {
        "step": "uma",
        "gen_path": gen_path,
        "outdir": outdir,
        "optimize": optimize_dict,
        "optimized": optimized,
        "md_metrics": metrics,
        "summary": summary,
    }


def _build_dft_result_from_job_request(req: Dict[str, Any]) -> Dict[str, Any]:
    outdir = req.get("outdir", DEMO_DEFAULT_OUTDIR)
    summary_path = str(Path(outdir) / "summary.json")
    summary = _find_json(Path(summary_path))
    dft = (summary or {}).get("dft") or {}
    verification = None
    claim = req.get("claim")
    if claim and dft:
        verification = _verify_claim_against_dft(
            claim,
            dft,
            gen_path=req.get("gen_path"),
            designer_client=req.get("designer_client"),
            api_key=req.get("api_key"),
        )

    return {
        "step": "dft",
        "outdir": outdir,
        "summary_path": summary_path if Path(summary_path).is_file() else None,
        "dft": dft,
        "summary": summary,
        "verification": verification,
    }


def _refresh_step_job_state(job_id: str):
    with DEMO_LOCK:
        job = DEMO_STEP_JOBS.get(job_id)
        proc = DEMO_STEP_PROCS.get(job_id)
    if not job:
        return

    log_path = job.get("log_path")
    if log_path and Path(log_path).is_file():
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            with DEMO_LOCK:
                if job_id in DEMO_STEP_JOBS:
                    DEMO_STEP_JOBS[job_id]["log"] = text[-200000:]
        except Exception:
            pass

    if job.get("status") in ("completed", "failed", "cancelled"):
        return
    if proc is None:
        return

    rc = proc.poll()
    if rc is None:
        with DEMO_LOCK:
            if job_id in DEMO_STEP_JOBS:
                DEMO_STEP_JOBS[job_id]["status"] = "running"
                DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()
        return

    if rc == 0:
        req = job.get("request") or {}
        kind = job.get("kind")
        if kind == "uma":
            result = _build_uma_result_from_job_request(req)
        elif kind == "dft":
            result = _build_dft_result_from_job_request(req)
        else:
            result = _build_optimize_result_from_job_request(req)
        with DEMO_LOCK:
            if job_id in DEMO_STEP_JOBS:
                DEMO_STEP_JOBS[job_id]["status"] = "completed"
                DEMO_STEP_JOBS[job_id]["result"] = result
                DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()
    else:
        with DEMO_LOCK:
            if job_id in DEMO_STEP_JOBS:
                DEMO_STEP_JOBS[job_id]["status"] = "failed"
                DEMO_STEP_JOBS[job_id]["error"] = {
                    "message": f"Optimization process exited with code {rc}",
                }
                DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()

    with DEMO_LOCK:
        DEMO_STEP_PROCS.pop(job_id, None)


def _run_optimize_step_sync(inp: DemoStepOptimizeIn) -> Dict[str, Any]:
    save_dir = _session_save_dir(inp.session_id)
    gen_path = inp.gen_path or str(_find_newest_npz(save_dir) or "")
    if not gen_path:
        raise HTTPException(status_code=400, detail="No gen_path provided and no generated structure found in default save dir.")

    outdir = _session_outdir(inp.session_id)
    opt_in = OptimizeIn(
        gen_path=gen_path,
        outdir=outdir,
        device=inp.device,
        preset=inp.preset,
        loops=inp.loops,
        run_dft=inp.run_dft,
        uma_ckpt=DEMO_DEFAULT_UMA_CKPT,
        uma_config_yml=None,
        orca_command=os.getenv("ORCA_COMMAND"),
        orca_maxcore=inp.orca_maxcore,
        orca_nprocs=inp.orca_nprocs,
        orca_simpleinput=None,
    )
    opt_out = optimize(opt_in)
    opt_dict = opt_out.model_dump() if hasattr(opt_out, "model_dump") else opt_out.dict()

    summary_path = opt_dict.get("summary_path")
    summary = _find_json(Path(summary_path)) if summary_path else None
    optimized = _atoms_to_json(opt_dict.get("uma", {}).get("traj")) if opt_dict.get("uma", {}).get("traj") else None
    metrics = _load_md_metrics(outdir)
    verification = None
    if inp.claim and summary and summary.get("dft"):
        verification = _verify_claim_against_dft(
            inp.claim,
            summary.get("dft") or {},
            gen_path=gen_path,
            designer_client=inp.designer_client,
            api_key=inp.api_key,
        )

    return {
        "step": "optimize",
        "gen_path": gen_path,
        "outdir": outdir,
        "optimize": opt_dict,
        "optimized": optimized,
        "md_metrics": metrics,
        "summary": summary,
        "verification": verification,
    }


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "optimizer_imported": HAS_OPTIMIZER}

@app.post("/generate", response_model=GenerateOut)
def generate(inp: GenerateIn):
    _ensure_exists(inp.ckpt_dir, "ckpt_dir")
    if inp.cif_dir and not inp.use_llm:
        # allowed but unused
        pass
    if inp.use_llm:
        _ensure_exists(inp.cif_dir or "", "cif_dir")

    gen_path, files, log = _run_gen_test(
        claims=inp.claims,
        ckpt_dir=inp.ckpt_dir,
        cif_dir=inp.cif_dir,
        save_dir=inp.save_dir,
        use_llm=inp.use_llm,
        prototype=inp.prototype,
        no_generator=inp.no_generator,
        api_key=inp.api_key,
        extra_args=inp.extra_args,
        designer_client=inp.designer_client,
    )
    return GenerateOut(gen_path=str(gen_path), saved_files=files, log=log)

@app.post("/optimize", response_model=OptimizeOut)
def optimize(inp: OptimizeIn):
    if not HAS_OPTIMIZER:
        raise HTTPException(status_code=500, detail=f"wrap_md_uma import failed: {_IMPORT_ERR}")

    _ensure_exists(inp.gen_path, "gen_path")
    Path(inp.outdir).mkdir(parents=True, exist_ok=True)

    # Allow env fallbacks
    uma_ckpt = inp.uma_ckpt or os.getenv("FAIRCHEM_UMA_CKPT")
    uma_cfg  = inp.uma_config_yml or os.getenv("FAIRCHEM_UMA_CONFIG")
    orca_cmd = inp.orca_command or os.getenv("ORCA_COMMAND")

    try:
        res = optimize_and_characterize(
            gen_path=inp.gen_path,
            outdir=inp.outdir,
            uma_ckpt=uma_ckpt,
            uma_config_yml=uma_cfg,
            device=inp.device,
            preset=inp.preset,
            loops=inp.loops,
            run_dft=inp.run_dft,
            orca_command=orca_cmd,
            orca_maxcore=inp.orca_maxcore,
            orca_nprocs=inp.orca_nprocs,
            orca_simpleinput=inp.orca_simpleinput,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"optimize_and_characterize failed: {e}") from e

    summary = Path(inp.outdir) / "summary.json"
    return OptimizeOut(
        uma=res.get("uma", {}),
        dft=res.get("dft"),
        summary_path=str(summary) if summary.exists() else None,
    )

@app.post("/pipeline", response_model=PipelineOut)
def pipeline(inp: PipelineIn):
    # Step 1: generate
    gen_path, _, _ = _run_gen_test(
        claims=inp.claims,
        ckpt_dir=inp.ckpt_dir,
        cif_dir=inp.cif_dir,
        save_dir=inp.save_dir,
        use_llm=inp.use_llm,
        prototype=inp.prototype,
        no_generator=inp.no_generator,
        api_key=inp.api_key,
        extra_args=inp.extra_args,
        designer_client=inp.designer_client,
    )

    # Step 2/3: optimize (optional)
    optimize_out: Optional[OptimizeOut] = None
    if HAS_OPTIMIZER:
        opt_in = OptimizeIn(
            gen_path=str(gen_path),
            outdir=inp.outdir,
            device=inp.device,
            preset=inp.preset,
            loops=inp.loops,
            run_dft=inp.run_dft,
            uma_ckpt=inp.uma_ckpt,
            uma_config_yml=inp.uma_config_yml,
            orca_command=inp.orca_command,
            orca_maxcore=inp.orca_maxcore,
            orca_nprocs=inp.orca_nprocs,
            orca_simpleinput=inp.orca_simpleinput,
        )
        optimize_out = optimize(opt_in)  # reuse handler
    return PipelineOut(gen_path=str(gen_path), optimize=optimize_out)


@app.post("/demo/run")
def demo_run(inp: DemoRunIn):
    return _start_pipeline_job(inp)


@app.get("/demo/defaults")
def demo_defaults():
    return {
        "model": DEMO_DEFAULT_MODEL,
        "save_dir": DEMO_DEFAULT_SAVE_DIR,
        "outdir": DEMO_DEFAULT_OUTDIR,
        "no_generator": True,
        "run_dft": True,
        "loops": 5,
        "orca_nprocs": 8,
        "orca_maxcore": 4000,
        "modes": ["llm", "rocksalt", "diamond"],
        "presets": ["quick", "sprint", "standard", "thorough"],
        "devices": ["cuda", "cpu"],
    }


@app.post("/demo/step/generate")
def demo_step_generate(inp: DemoStepGenerateIn):
    mode_args = _step_mode_to_generate_args(inp.mode)
    save_dir = _session_save_dir(inp.session_id)
    with GPU_SEMAPHORE:
        gen_path, files, log = _run_gen_test(
            claims=inp.claims,
            ckpt_dir=DEMO_DEFAULT_CKPT_DIR,
            cif_dir=None,
            save_dir=save_dir,
            use_llm=mode_args["use_llm"],
            prototype=mode_args["prototype"],
            no_generator=inp.no_generator,
            api_key=inp.api_key,
            extra_args=None,
            designer_client=inp.model or DEMO_DEFAULT_MODEL,
        )
    generated = _atoms_to_json(str(gen_path))
    return {
        "step": "generate",
        "gen_path": str(gen_path),
        "generated": generated,
        "saved_files": files,
        "no_generator": inp.no_generator,
        "log": log,
    }


@app.post("/demo/step/optimize")
def demo_step_optimize(inp: DemoStepOptimizeIn):
    return _run_optimize_step_sync(inp)


@app.post("/demo/step/optimize/start")
def demo_step_optimize_start(inp: DemoStepOptimizeIn):
    save_dir = _session_save_dir(inp.session_id)
    gen_path = inp.gen_path or str(_find_newest_npz(save_dir) or "")
    if not gen_path:
        raise HTTPException(status_code=400, detail="No gen_path provided and no generated structure found in default save dir.")

    outdir = _session_outdir(inp.session_id)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    payload = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    payload["gen_path"] = gen_path
    payload["outdir"] = outdir

    job_id = _create_step_job("uma", payload, session_id=inp.session_id)
    log_path = str(Path(outdir) / f"demo_opt_{job_id}.log")
    cmd = [
        sys.executable,
        "wrap_md_uma.py",
        "--gen-path", gen_path,
        "--outdir", outdir,
        "--device", inp.device,
        "--preset", inp.preset,
        "--loops", str(inp.loops),
        "--nprocs", str(inp.orca_nprocs),
        "--maxcore", str(inp.orca_maxcore),
        "--ckpt", DEMO_DEFAULT_UMA_CKPT,
    ]
    # UMA-only phase in Step 2A

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
        )

    with DEMO_LOCK:
        if job_id in DEMO_STEP_JOBS:
            DEMO_STEP_JOBS[job_id]["status"] = "running"
            DEMO_STEP_JOBS[job_id]["log_path"] = log_path
            DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()
        DEMO_STEP_PROCS[job_id] = proc

    return {"job_id": job_id, "status": "running"}


@app.post("/demo/step/dft/start")
def demo_step_dft_start(inp: DemoStepDftIn):
    outdir = _session_outdir(inp.session_id) if inp.session_id else (inp.outdir or DEMO_DEFAULT_OUTDIR)
    best_traj = Path(outdir) / "best.traj"
    if not best_traj.is_file():
        raise HTTPException(status_code=400, detail=f"Missing best.traj for DFT step: {best_traj}")

    payload = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    payload["outdir"] = outdir

    job_id = _create_step_job("dft", payload, session_id=inp.session_id)
    log_path = str(Path(outdir) / f"demo_dft_{job_id}.log")
    orca_cmd = os.getenv("ORCA_COMMAND", "orca")

    py_code = (
        "import json, os; "
        "from ase.io import read; "
        "from wrap_md_uma import run_orca_dft; "
        f"outdir={outdir!r}; "
        "best=os.path.join(outdir,'best.traj'); "
        "orca_dir=os.path.join(outdir,'orca_sp'); "
        "atoms=read(best); "
        f"dft, _ = run_orca_dft(atoms, orca_command={orca_cmd!r}, workdir=orca_dir, maxcore={int(inp.orca_maxcore)}, nprocs={int(inp.orca_nprocs)}, timer_label='ORCA DFT Single-Point', save_dir=orca_dir); "
        "open(os.path.join(orca_dir,'dft_results.json'),'w').write(json.dumps(dft,indent=2)); "
        "summary_path=os.path.join(outdir,'summary.json'); "
        "summary={}; "
        "\ntry:\n summary=json.load(open(summary_path,'r'))\nexcept Exception:\n pass\n"
        "summary['dft']=dft; "
        "open(summary_path,'w').write(json.dumps(summary,indent=2)); "
        "print(json.dumps({'status':'ok','summary_path':summary_path}, indent=2))"
    )

    cmd = [sys.executable, "-u", "-c", py_code]
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
        )

    with DEMO_LOCK:
        if job_id in DEMO_STEP_JOBS:
            DEMO_STEP_JOBS[job_id]["status"] = "running"
            DEMO_STEP_JOBS[job_id]["log_path"] = log_path
            DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()
        DEMO_STEP_PROCS[job_id] = proc

    return {"job_id": job_id, "status": "running"}


@app.get("/demo/step/jobs/{job_id}", response_model=DemoStepJobOut)
def demo_step_job(job_id: str):
    _refresh_step_job_state(job_id)
    job = _get_step_job(job_id)
    return DemoStepJobOut(
        job_id=job["job_id"],
        status=job["status"],
        log=job.get("log", ""),
        result=job.get("result"),
        error=job.get("error"),
    )


@app.post("/demo/step/jobs/{job_id}/stop")
def demo_step_job_stop(job_id: str):
    _refresh_step_job_state(job_id)
    with DEMO_LOCK:
        job = DEMO_STEP_JOBS.get(job_id)
        proc = DEMO_STEP_PROCS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Step job not found: {job_id}")

    if job.get("status") in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": job.get("status"), "message": "Job already finished."}

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)

    with DEMO_LOCK:
        if job_id in DEMO_STEP_JOBS:
            DEMO_STEP_JOBS[job_id]["status"] = "cancelled"
            DEMO_STEP_JOBS[job_id]["error"] = {"message": "Stopped by user."}
            DEMO_STEP_JOBS[job_id]["updated_at"] = time.time()
        DEMO_STEP_PROCS.pop(job_id, None)

    _refresh_step_job_state(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/demo/step/jobs/stop-all")
def demo_step_jobs_stop_all(inp: DemoStopAllIn = DemoStopAllIn()):
    stopped = []
    session_id = inp.session_id
    with DEMO_LOCK:
        job_ids = list(DEMO_STEP_JOBS.keys())

    for jid in job_ids:
        _refresh_step_job_state(jid)
        with DEMO_LOCK:
            job = DEMO_STEP_JOBS.get(jid)
            proc = DEMO_STEP_PROCS.get(jid)
        if not job:
            continue
        # Only stop jobs belonging to this session (or all if no session_id)
        if session_id and job.get("session_id") != session_id:
            continue
        if job.get("status") in ("completed", "failed", "cancelled"):
            continue

        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

        with DEMO_LOCK:
            if jid in DEMO_STEP_JOBS:
                DEMO_STEP_JOBS[jid]["status"] = "cancelled"
                DEMO_STEP_JOBS[jid]["error"] = {"message": "Stopped by user (stop-all)."}
                DEMO_STEP_JOBS[jid]["updated_at"] = time.time()
            DEMO_STEP_PROCS.pop(jid, None)
        stopped.append(jid)

    return {"status": "cancelled", "stopped_jobs": stopped, "count": len(stopped)}


@app.post("/demo/step/verify", response_model=DemoVerifyOut)
def demo_step_verify(inp: DemoStepVerifyIn):
    outdir = _session_outdir(inp.session_id)
    summary_path = inp.summary_path or str(Path(outdir) / "summary.json")
    summ = _find_json(Path(summary_path))
    if not summ:
        raise HTTPException(status_code=400, detail=f"Summary not found: {summary_path}")
    dft = summ.get("dft") or {}
    if not dft:
        raise HTTPException(status_code=400, detail="No DFT fields found in summary.")
    result = _verify_claim_against_dft(
        inp.claim,
        dft,
        gen_path=inp.gen_path,
        designer_client=inp.designer_client,
        api_key=inp.api_key,
    )
    return DemoVerifyOut(**result)


@app.get("/demo/jobs/{job_id}")
def demo_job(job_id: str):
    return _get_job(job_id)


@app.get("/demo/jobs")
def demo_jobs(limit: int = 20):
    with DEMO_LOCK:
        all_jobs = list(DEMO_JOBS.values())
    all_jobs = sorted(all_jobs, key=lambda x: x.get("created_at", 0), reverse=True)
    return {"jobs": all_jobs[: max(1, min(limit, 100))]}


@app.get("/demo/structure")
def demo_structure(path: str):
    return _atoms_to_json(path)


@app.get("/demo/md-metrics")
def demo_md_metrics(outdir: str):
    return _load_md_metrics(outdir)


@app.get("/demo/dft")
def demo_dft(summary_path: str):
    summ = _find_json(Path(summary_path))
    if summ is None:
        raise HTTPException(status_code=404, detail=f"summary not found or invalid: {summary_path}")
    return {"dft": summ.get("dft") or {}, "uma": summ.get("uma") or {}, "summary": summ}


@app.post("/demo/verify", response_model=DemoVerifyOut)
def demo_verify(inp: DemoVerifyIn):
    dft_data = inp.dft or {}
    if not dft_data and inp.summary_path:
        summ = _find_json(Path(inp.summary_path))
        if summ:
            dft_data = summ.get("dft") or {}

    if not dft_data:
        raise HTTPException(status_code=400, detail="No DFT data provided. Pass dft or summary_path.")

    result = _verify_claim_against_dft(
        inp.claim,
        dft_data,
        gen_path=inp.gen_path,
        designer_client=inp.designer_client,
        api_key=inp.api_key,
    )
    return DemoVerifyOut(**result)


@app.get("/demo/jobs/{job_id}/bundle")
def demo_job_bundle(job_id: str):
    job = _get_job(job_id)
    if job.get("status") != "completed":
        return {"job": job, "bundle": None}

    req = job.get("request") or {}
    res = job.get("result") or {}
    gen_path = res.get("gen_path")
    opt = (res.get("optimize") or {})
    uma = (opt.get("uma") or {})
    summary_path = opt.get("summary_path")
    outdir = req.get("outdir")

    generated = _atoms_to_json(gen_path) if gen_path else None
    optimized = _atoms_to_json(uma.get("traj")) if uma.get("traj") else None
    metrics = _load_md_metrics(outdir) if outdir else None
    summary = _find_json(Path(summary_path)) if summary_path else None
    dft = (summary or {}).get("dft") or (opt.get("dft") or {})
    verification = _verify_claim_against_dft(
        req.get("claims", ""),
        dft,
        gen_path=gen_path,
        designer_client=req.get("designer_client"),
        api_key=req.get("api_key"),
    ) if dft else None

    return {
        "job": job,
        "bundle": {
            "claim": req.get("claims"),
            "generated": generated,
            "optimized": optimized,
            "md_metrics": metrics,
            "summary": summary,
            "verification": verification,
        },
    }


if DEMO_ROOT.is_dir():
    app.mount("/demo-assets", StaticFiles(directory=str(DEMO_ROOT)), name="demo-assets")


@app.get("/demo")
def demo_index():
    index_path = DEMO_ROOT / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Demo UI not found. Expected web_demo/index.html")
    return FileResponse(str(index_path))
