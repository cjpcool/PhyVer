#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
from ase.io import read as ase_read

from batch_claim_pipeline import parse_generation_time, parse_orca_time, run_cmd
from claim_verification_llm import verify_claim_with_llm
from wrap_md_uma import run_orca_dft


LIKERT_LEVELS = [-2, -1, 0, 1, 2]

DEMO_DEFAULT_CKPT_DIR = os.getenv("DEMO_CKPT_DIR", "./checkpoints/omat24_rattle2")
DEMO_DEFAULT_UMA_CKPT = os.getenv("FAIRCHEM_UMA_CKPT", "./checkpoints/uma-s-1p1.pt")
DEMO_DEFAULT_MODEL = os.getenv("DEMO_DESIGNER_MODEL", "gpt-5.1")


def read_gold_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "claim" not in obj or "likert_score" not in obj:
                continue
            rows.append(
                {
                    "line_no": line_no,
                    "problem_id": obj.get("problem_id"),
                    "claim": str(obj.get("claim", "")).strip(),
                    "gold_score": int(obj.get("likert_score")),
                    "domain": obj.get("domain"),
                    "subdomain": obj.get("subdomain"),
                }
            )
    return rows


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_error_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"[{ts}] problem_id={row.get('problem_id')} status={row.get('status')} "
            f"pipeline_error={row.get('pipeline_error')} verify_error={row.get('verify_error')}\n"
        )
        llm_raw = row.get("llm_raw")
        if isinstance(llm_raw, str) and llm_raw.strip():
            llm_raw_preview = llm_raw[:500].replace("\n", "\\n")
            f.write(f"  llm_raw_preview={llm_raw_preview}\n")


def write_confusion_csv(path: Path, confusion: Dict[int, Dict[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("gold\\pred," + ",".join(str(x) for x in LIKERT_LEVELS) + "\n")
        for g in LIKERT_LEVELS:
            row = [str(confusion.get(g, {}).get(p, 0)) for p in LIKERT_LEVELS]
            f.write(f"{g}," + ",".join(row) + "\n")


def _safe_mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if isinstance(r.get("pred_score"), int)]
    total = len(rows)
    n_scored = len(usable)

    metrics: Dict[str, Any] = {
        "n_total": total,
        "n_scored": n_scored,
        "coverage": (n_scored / total) if total else 0.0,
    }
    if not usable:
        return metrics

    gold = [int(r["gold_score"]) for r in usable]
    pred = [int(r["pred_score"]) for r in usable]

    exact = [int(p == g) for p, g in zip(pred, gold)]
    within1 = [int(abs(p - g) <= 1) for p, g in zip(pred, gold)]
    abs_err = [abs(p - g) for p, g in zip(pred, gold)]
    sq_err = [(p - g) ** 2 for p, g in zip(pred, gold)]

    metrics.update(
        {
            "accuracy_exact": _safe_mean(exact),
            "accuracy_within_1": _safe_mean(within1),
            "mae": _safe_mean(abs_err),
            "rmse": math.sqrt(_safe_mean(sq_err) or 0.0),
        }
    )

    confusion: Dict[int, Dict[int, int]] = {
        g: {p: 0 for p in LIKERT_LEVELS} for g in LIKERT_LEVELS
    }
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    metrics["confusion_matrix"] = confusion

    per_class: Dict[int, Dict[str, Optional[float]]] = {}
    f1_vals: List[float] = []
    for cls in LIKERT_LEVELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        support = sum(1 for g in gold if g == cls)

        prec = (tp / (tp + fp)) if (tp + fp) else None
        rec = (tp / (tp + fn)) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
        if f1 is not None:
            f1_vals.append(f1)

        per_class[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

    metrics["per_class"] = per_class
    metrics["macro_f1"] = _safe_mean(f1_vals)
    return metrics


def load_generated_structure(npz_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not npz_path:
        return None
    p = Path(npz_path)
    if not p.is_file():
        return None

    try:
        data = np.load(p, allow_pickle=False)
        atom_types = data["atom_types"].astype(int).tolist()
        coords = data["cart_coords"].astype(float)
        lengths = data["lengths"].astype(float).reshape(-1).tolist()
        angles = data["angles"].astype(float).reshape(-1).tolist()
        comp = dict(Counter(atom_types))
        atom_preview = []
        n_preview = min(8, len(atom_types))
        for i in range(n_preview):
            atom_preview.append(
                {
                    "Z": int(atom_types[i]),
                    "position": [float(x) for x in coords[i].tolist()],
                }
            )
        return {
            "available": True,
            "source": "generated_npz",
            "format": "npz",
            "n_atoms": len(atom_types),
            "lengths": lengths,
            "angles": angles,
            "composition": comp,
            "atom_preview": atom_preview,
        }
    except Exception:
        return None


def load_dft_dict(uma_summary_path: Optional[str], energies: Dict[str, Any]) -> Dict[str, Any]:
    dft: Dict[str, Any] = {}
    if uma_summary_path and Path(uma_summary_path).is_file():
        try:
            summary = json.loads(Path(uma_summary_path).read_text(encoding="utf-8"))
            if isinstance(summary, dict) and isinstance(summary.get("dft"), dict):
                dft = dict(summary["dft"])
        except Exception:
            pass

    if not dft:
        dft = {
            "energy_eV": energies.get("dft_energy_eV"),
            "gap_eV": energies.get("dft_gap_eV"),
        }
    return dft


def build_pipeline_args(args: argparse.Namespace, output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=str(output_root),
        ckpt_gen=args.ckpt_gen,
        designer_client=args.verify_model,
        logic_mode=args.logic_mode,
        cutoff=args.cutoff,
        api_key=args.api_key or "",
        ase_omat=args.ase_omat,
        no_llm=args.no_llm,
        prototype=args.prototype,
        no_generator=args.no_generator,
        uma_ckpt=args.uma_ckpt,
        device=args.device,
        relax_cell=args.relax_cell,
        fallback_emt=args.fallback_emt,
        dft=args.dft,
        orca_command=args.orca_command,
        nprocs=args.nprocs,
        maxcore=args.maxcore,
        orcasimpleinput=args.orcasimpleinput,
        orca_extra_blocks=args.orca_extra_blocks,
        preset=args.preset,
        loops=args.loops,
        fmax=args.fmax,
    )


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _claim_paths(output_root: str, idx: int) -> Dict[str, Path]:
    claim_dir = Path(output_root) / f"claim_{idx:04d}"
    return {
        "claim_dir": claim_dir,
        "gen_dir": claim_dir / "gen",
        "md_dir": claim_dir / "mdopt",
        "analysis_dir": claim_dir / "analysis",
    }


def _latest_npz(gen_dir: Path) -> Optional[Path]:
    if not gen_dir.is_dir():
        return None
    candidates = sorted(gen_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _is_uma_done(md_dir: Path) -> bool:
    summary = _read_json(md_dir / "summary.json") or {}
    return isinstance(summary.get("uma"), dict) and (summary.get("uma", {}).get("energy_eV") is not None)


def _is_dft_done(md_dir: Path) -> bool:
    summary = _read_json(md_dir / "summary.json") or {}
    return isinstance(summary.get("dft"), dict) and (summary.get("dft", {}).get("energy_eV") is not None)


def _is_analysis_done(analysis_dir: Path) -> bool:
    return (analysis_dir / "metrics.json").is_file()


def _run_generation_step(idx: int, claim: str, pipe_args: SimpleNamespace) -> Dict[str, Any]:
    paths = _claim_paths(pipe_args.output_root, idx)
    claim_dir, gen_dir = paths["claim_dir"], paths["gen_dir"]
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / "raw_claim.txt").write_text(claim, encoding="utf-8")

    gen_npz = _latest_npz(gen_dir)
    if gen_npz is not None:
        return {
            "rc": 0,
            "wall": 0.0,
            "gen_logged": parse_generation_time(gen_dir) or 0.0,
            "gen_npz": gen_npz,
            "skipped": True,
        }

    gen_cmd = [
        os.sys.executable,
        "gen_test.py",
        "--prompt",
        claim,
        "--save-dir",
        str(gen_dir),
        "--ckpt-dir",
        pipe_args.ckpt_gen,
        "--logic-mode",
        pipe_args.logic_mode,
        "--cutoff",
        str(pipe_args.cutoff),
    ]
    if pipe_args.api_key:
        gen_cmd += ["--api-key", pipe_args.api_key]
    else:
        gen_cmd += ["--api-key", ""]
    if getattr(pipe_args, "designer_client", None):
        gen_cmd += ["--designer-client", pipe_args.designer_client]
    if pipe_args.ase_omat:
        gen_cmd += ["--ase-omat", pipe_args.ase_omat]
    if pipe_args.no_llm:
        gen_cmd += ["--no-llm"]
    if pipe_args.prototype:
        gen_cmd += ["--prototype", pipe_args.prototype]
    if pipe_args.no_generator:
        gen_cmd += ["--no-generator"]

    rc, wall, _ = run_cmd(gen_cmd, log_path=gen_dir / "gen_stdout.log")
    gen_logged = parse_generation_time(gen_dir) or wall
    gen_npz = _latest_npz(gen_dir)
    return {"rc": rc, "wall": wall, "gen_logged": gen_logged, "gen_npz": gen_npz, "skipped": False}


def _run_uma_step(idx: int, gen_npz: Path, pipe_args: SimpleNamespace) -> Dict[str, Any]:
    paths = _claim_paths(pipe_args.output_root, idx)
    md_dir = paths["md_dir"]
    if _is_uma_done(md_dir):
        return {"rc": 0, "wall": 0.0, "skipped": True}

    uma_cmd = [
        os.sys.executable,
        "wrap_md_uma.py",
        "--gen-path",
        str(gen_npz),
        "--outdir",
        str(md_dir),
        "--preset",
        pipe_args.preset,
        "--loops",
        str(pipe_args.loops),
        "--fmax",
        str(pipe_args.fmax),
    ]
    if pipe_args.uma_ckpt:
        uma_cmd += ["--ckpt", pipe_args.uma_ckpt]
    if pipe_args.device:
        uma_cmd += ["--device", pipe_args.device]
    if pipe_args.relax_cell:
        uma_cmd += ["--relax-cell"]
    if pipe_args.fallback_emt:
        uma_cmd += ["--fallback-emt"]

    rc, wall, _ = run_cmd(uma_cmd, log_path=md_dir / "uma_stdout.log")
    return {"rc": rc, "wall": wall, "skipped": False}


def _run_dft_step(idx: int, pipe_args: SimpleNamespace) -> Dict[str, Any]:
    paths = _claim_paths(pipe_args.output_root, idx)
    md_dir = paths["md_dir"]
    if not pipe_args.dft:
        return {"rc": 0, "wall": 0.0, "skipped": True}
    if _is_dft_done(md_dir):
        return {"rc": 0, "wall": 0.0, "skipped": True}

    best_traj = md_dir / "best.traj"
    if not best_traj.is_file():
        return {"rc": 1, "wall": 0.0, "error": "best.traj missing before DFT", "skipped": False}

    try:
        atoms = ase_read(best_traj)
        orca_dir = md_dir / "orca_sp"
        (dft, elapsed) = run_orca_dft(
            atoms,
            orca_command=pipe_args.orca_command or "orca",
            workdir=str(orca_dir),
            maxcore=pipe_args.maxcore,
            nprocs=pipe_args.nprocs,
            orcasimpleinput=pipe_args.orcasimpleinput,
            extra_blocks=pipe_args.orca_extra_blocks or "",
            timer_label="ORCA DFT Single-Point",
            save_dir=str(orca_dir),
        )
        (orca_dir / "dft_results.json").write_text(json.dumps(dft, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_path = md_dir / "summary.json"
        summary = _read_json(summary_path) or {}
        if "uma" not in summary:
            summary["uma"] = {"energy_eV": None, "traj": str(best_traj), "xyz": str(md_dir / "best.xyz")}
        summary["dft"] = dft
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"rc": 0, "wall": float(elapsed), "skipped": False}
    except Exception as e:
        return {"rc": 1, "wall": 0.0, "error": str(e), "skipped": False}


def _run_analysis_step(idx: int, pipe_args: SimpleNamespace) -> Dict[str, Any]:
    paths = _claim_paths(pipe_args.output_root, idx)
    md_dir, analysis_dir = paths["md_dir"], paths["analysis_dir"]
    if _is_analysis_done(analysis_dir):
        return {"rc": 0, "wall": 0.0, "skipped": True}

    analysis_cmd = [
        os.sys.executable,
        "structure_optim_modules/analyze_optimization.py",
        "--root",
        str(md_dir),
        "--pattern",
        "loop_*.traj",
        "--save-dir",
        str(analysis_dir),
        "--export-json",
        str(analysis_dir / "metrics.json"),
        "--export-csv",
        str(analysis_dir / "metrics.csv"),
        "--no-show",
        "--compare",
    ]
    rc, wall, _ = run_cmd(analysis_cmd, log_path=analysis_dir / "analysis_stdout.log")
    return {"rc": rc, "wall": wall, "skipped": False}


def _collect_pipeline_result(idx: int, claim: str, pipe_args: SimpleNamespace, stage_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    paths = _claim_paths(pipe_args.output_root, idx)
    gen_npz = _latest_npz(paths["gen_dir"])
    uma_summary = paths["md_dir"] / "summary.json"

    uma_energy = None
    dft_energy = None
    dft_gap = None
    if uma_summary.is_file():
        data = _read_json(uma_summary) or {}
        uma_energy = (data.get("uma") or {}).get("energy_eV")
        if isinstance(data.get("dft"), dict):
            dft_energy = data["dft"].get("energy_eV")
            dft_gap = data["dft"].get("gap_eV")

    return {
        "index": idx,
        "claim": claim,
        "paths": {
            "claim_dir": str(paths["claim_dir"]),
            "gen_dir": str(paths["gen_dir"]),
            "md_dir": str(paths["md_dir"]),
            "analysis_dir": str(paths["analysis_dir"]),
            "generated_npz": str(gen_npz) if gen_npz else None,
            "uma_summary": str(uma_summary) if uma_summary.is_file() else None,
        },
        "return_codes": {
            "gen": stage_stats["gen"].get("rc", 1),
            "uma": stage_stats["uma"].get("rc", 1),
            "dft": stage_stats["dft"].get("rc", 1),
            "analysis": stage_stats["analysis"].get("rc", 1),
        },
        "timing_seconds": {
            "gen_wall": stage_stats["gen"].get("wall", 0.0),
            "gen_logged": stage_stats["gen"].get("gen_logged", stage_stats["gen"].get("wall", 0.0)),
            "uma_wall": stage_stats["uma"].get("wall", 0.0),
            "dft_logged": parse_orca_time(paths["md_dir"]) if pipe_args.dft else None,
            "analysis_wall": stage_stats["analysis"].get("wall", 0.0),
        },
        "energies": {
            "uma_energy_eV": uma_energy,
            "dft_energy_eV": dft_energy,
            "dft_gap_eV": dft_gap,
        },
    }


def _run_single_claim_with_retry(
    *,
    idx: int,
    row: Dict[str, Any],
    pipe_args_dict: Dict[str, Any],
    verify_model: str,
    api_key: Optional[str],
    max_attempts: int,
) -> Dict[str, Any]:
    rec = dict(row)
    rec["index"] = idx
    rec["pipeline_status"] = "unknown"
    rec["status"] = "error"

    t0_all = time.time()
    pipe_args = SimpleNamespace(**pipe_args_dict)
    stage_attempts = {"gen": 0, "uma": 0, "dft": 0, "analysis": 0, "verify": 0}
    stage_stats: Dict[str, Dict[str, Any]] = {
        "gen": {"rc": 1, "wall": 0.0},
        "uma": {"rc": 1, "wall": 0.0},
        "dft": {"rc": 0 if not pipe_args.dft else 1, "wall": 0.0},
        "analysis": {"rc": 1, "wall": 0.0},
    }

    pipeline_error: Optional[str] = None
    verify_error: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None

    # Stage 1: generation
    gen_result: Dict[str, Any] = {}
    for _ in range(max_attempts):
        stage_attempts["gen"] += 1
        gen_result = _run_generation_step(idx, row["claim"], pipe_args)
        stage_stats["gen"] = gen_result
        if gen_result.get("rc", 1) == 0 and gen_result.get("gen_npz") is not None:
            break
    gen_npz = gen_result.get("gen_npz")
    if stage_stats["gen"].get("rc", 1) != 0 or gen_npz is None:
        pipeline_error = "Generation failed after retries"

    # Stage 2: UMA (only if gen ok)
    if pipeline_error is None:
        uma_result: Dict[str, Any] = {}
        for _ in range(max_attempts):
            stage_attempts["uma"] += 1
            uma_result = _run_uma_step(idx, gen_npz, pipe_args)
            stage_stats["uma"] = uma_result
            if uma_result.get("rc", 1) == 0:
                break
        if stage_stats["uma"].get("rc", 1) != 0:
            pipeline_error = "UMA failed after retries"

    # Stage 3: DFT (optional)
    if pipeline_error is None and pipe_args.dft:
        dft_result: Dict[str, Any] = {}
        for _ in range(max_attempts):
            stage_attempts["dft"] += 1
            dft_result = _run_dft_step(idx, pipe_args)
            stage_stats["dft"] = dft_result
            if dft_result.get("rc", 1) == 0:
                break
        if stage_stats["dft"].get("rc", 1) != 0:
            pipeline_error = "DFT failed after retries"

    # Stage 4: analysis
    if pipeline_error is None:
        ana_result: Dict[str, Any] = {}
        for _ in range(max_attempts):
            stage_attempts["analysis"] += 1
            ana_result = _run_analysis_step(idx, pipe_args)
            stage_stats["analysis"] = ana_result
            if ana_result.get("rc", 1) == 0:
                break
        if stage_stats["analysis"].get("rc", 1) != 0:
            pipeline_error = "Analysis failed after retries"

    process_res = _collect_pipeline_result(idx, row["claim"], pipe_args, stage_stats)

    if pipeline_error is None:
        dft = load_dft_dict(process_res["paths"].get("uma_summary"), process_res.get("energies", {}))
        structure_ctx = load_generated_structure(process_res["paths"].get("generated_npz"))
        for _ in range(max_attempts):
            stage_attempts["verify"] += 1
            try:
                verification = verify_claim_with_llm(
                    claim=row["claim"],
                    generated_structure=structure_ctx,
                    dft=dft,
                    designer_client=verify_model,
                    api_key=api_key,
                )
                break
            except Exception as e:
                verify_error = str(e)

    rec.update(
        {
            "pipeline_paths": process_res.get("paths"),
            "pipeline_return_codes": process_res.get("return_codes"),
            "pipeline_timing_seconds": process_res.get("timing_seconds"),
            "pipeline_energies": process_res.get("energies"),
            "stage_attempts": stage_attempts,
            "max_attempts": max_attempts,
        }
    )

    if verification is not None:
        rec.update(
            {
                "pred_score": verification.get("score"),
                "pred_reason": verification.get("reason"),
                "model_used": verification.get("model_used") or verify_model,
                "status": "ok" if isinstance(verification.get("score"), int) else "error",
                "llm_raw": verification.get("llm_raw"),
                "verification_verdict": verification.get("verdict"),
                "verification_checks": verification.get("checks"),
                "abs_error": abs(int(verification.get("score")) - int(row["gold_score"])) if isinstance(verification.get("score"), int) else None,
            }
        )
    else:
        rec.update(
            {
                "pred_score": None,
                "pred_reason": None,
                "model_used": verify_model,
                "status": "error",
            }
        )

    rec["pipeline_status"] = "ok" if pipeline_error is None else "error"
    if rec["pipeline_status"] == "error":
        rec["status"] = "error"

    rec["pipeline_error"] = pipeline_error
    rec["verify_error"] = verify_error
    rec["elapsed_total_sec"] = round(time.time() - t0_all, 3)
    return rec


def _load_latest_records(predictions_path: Path) -> Dict[int, Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    if not predictions_path.is_file():
        return latest
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = rec.get("index")
            if isinstance(idx, int):
                latest[idx] = rec
    return latest


def _is_record_completed(rec: Optional[Dict[str, Any]], require_dft: bool) -> bool:
    if not isinstance(rec, dict):
        return False
    if rec.get("status") != "ok" or rec.get("pipeline_status") != "ok":
        return False
    if not isinstance(rec.get("pred_score"), int):
        return False
    if require_dft:
        dft_e = (rec.get("pipeline_energies") or {}).get("dft_energy_eV")
        if dft_e is None:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch run PhyVer pipeline and evaluate Likert scores against gold labels.")
    ap.add_argument("--gold-jsonl", required=True, help="Path to a gold JSONL evaluation file.")
    ap.add_argument("--outdir", default=None, help="Default: ./artifacts/batch_runs/phyver_eval_<timestamp>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes for batch claims")
    ap.add_argument("--max-attempts", type=int, default=3, help="Retry attempts per claim on error (1-3)")

    ap.add_argument("--verify-model", default=DEMO_DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None, help="Used by both generation LLM and verifier LLM if needed.")

    ap.add_argument("--ckpt-gen", default=DEMO_DEFAULT_CKPT_DIR)
    ap.add_argument("--ase-omat", default=None)
    ap.add_argument("--logic-mode", default="union")
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--prototype", default=None)
    ap.add_argument("--no-generator", action="store_true")

    ap.add_argument("--uma-ckpt", default=DEMO_DEFAULT_UMA_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--relax-cell", action="store_true")
    ap.add_argument("--fallback-emt", action="store_true")
    ap.add_argument("--dft", action="store_true", default=False)
    ap.add_argument("--no-dft", action="store_true", help="Disable DFT step.")
    ap.add_argument("--orca-command", default=None)
    ap.add_argument("--nprocs", type=int, default=8)
    ap.add_argument("--maxcore", type=int, default=4000)
    ap.add_argument("--orcasimpleinput", default=None)
    ap.add_argument("--orca-extra-blocks", dest="orca_extra_blocks", default=None)

    ap.add_argument("--preset", default="quick", choices=["quick", "standard", "thorough"])
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--fmax", type=float, default=0.03)
    args = ap.parse_args()

    if args.no_dft:
        args.dft = False

    args.max_attempts = max(1, min(3, int(args.max_attempts)))
    args.workers = max(1, int(args.workers))

    gold_path = Path(args.gold_jsonl).resolve()
    rows = read_gold_jsonl(gold_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No valid claim rows found in: {gold_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).resolve() if args.outdir else (Path("./artifacts/batch_runs") / f"phyver_eval_{stamp}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline_root = outdir / "pipeline_runs"
    pipeline_root.mkdir(parents=True, exist_ok=True)

    predictions_path = outdir / "predictions.jsonl"
    errors_path = outdir / "errors.log"
    progress_path = outdir / "progress.json"
    if not predictions_path.exists():
        predictions_path.write_text("", encoding="utf-8")
    if not errors_path.exists():
        errors_path.write_text("", encoding="utf-8")

    pipe_args = build_pipeline_args(args, pipeline_root)
    pipe_args_dict = vars(pipe_args)

    existing_latest = _load_latest_records(predictions_path)
    indexed_rows = list(enumerate(rows, args.start_index))
    pending: List[tuple[int, Dict[str, Any]]] = []
    for i, row in indexed_rows:
        rec_old = existing_latest.get(i)
        if _is_record_completed(rec_old, require_dft=bool(args.dft)):
            print(f"[SKIP] idx={i:04d} problem_id={row.get('problem_id')} already completed", flush=True)
            continue
        pending.append((i, row))

    print(f"[RESUME] total={len(rows)} completed={len(rows)-len(pending)} pending={len(pending)}", flush=True)

    predictions_map: Dict[int, Dict[str, Any]] = dict(existing_latest)
    interrupted = False
    try:
        processed_count = len(rows) - len(pending)

        if args.workers == 1:
            for i, row in pending:
                print(f"[SUBMIT] idx={i:04d} problem_id={row.get('problem_id')} mode=single-worker", flush=True)
                rec = _run_single_claim_with_retry(
                    idx=i,
                    row=row,
                    pipe_args_dict=pipe_args_dict,
                    verify_model=args.verify_model,
                    api_key=args.api_key,
                    max_attempts=args.max_attempts,
                )
                processed_count += 1
                predictions_map[i] = rec
                append_jsonl_row(predictions_path, rec)
                if rec.get("status") == "error" or rec.get("pipeline_status") == "error":
                    append_error_log(errors_path, rec)

                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "processed": processed_count,
                            "total": len(rows),
                            "status": "running",
                            "last_problem_id": row.get("problem_id"),
                        },
                        pf,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    f"[{processed_count:02d}/{len(rows)}] {row.get('problem_id')} "
                    f"pipeline={rec.get('pipeline_status')} pred={rec.get('pred_score')} "
                    f"status={rec.get('status')} attempts={rec.get('attempts_used')}"
                )

                if args.fail_fast and rec.get("status") == "error":
                    print("[FATAL] Fail-fast enabled. Stopping.")
                    break
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
                for i, row in pending:
                    print(f"[SUBMIT] idx={i:04d} problem_id={row.get('problem_id')} mode=process-pool workers={args.workers}", flush=True)
                    fut = executor.submit(
                        _run_single_claim_with_retry,
                        idx=i,
                        row=row,
                        pipe_args_dict=pipe_args_dict,
                        verify_model=args.verify_model,
                        api_key=args.api_key,
                        max_attempts=args.max_attempts,
                    )
                    future_map[fut] = row

                for fut in concurrent.futures.as_completed(future_map):
                    row = future_map[fut]
                    rec = fut.result()
                    processed_count += 1
                    idx = rec.get("index")
                    if isinstance(idx, int):
                        predictions_map[idx] = rec
                    append_jsonl_row(predictions_path, rec)
                    if rec.get("status") == "error" or rec.get("pipeline_status") == "error":
                        append_error_log(errors_path, rec)

                    with open(progress_path, "w", encoding="utf-8") as pf:
                        json.dump(
                            {
                                "processed": processed_count,
                                "total": len(rows),
                                "status": "running",
                                "last_problem_id": row.get("problem_id"),
                            },
                            pf,
                            ensure_ascii=False,
                            indent=2,
                        )

                    print(
                        f"[{processed_count:02d}/{len(rows)}] {row.get('problem_id')} "
                        f"pipeline={rec.get('pipeline_status')} pred={rec.get('pred_score')} "
                        f"status={rec.get('status')} attempts={rec.get('attempts_used')}"
                    )

                    if args.fail_fast and rec.get("status") == "error":
                        print("[FATAL] Fail-fast enabled. Cancelling pending tasks.")
                        for f_pending in future_map:
                            if not f_pending.done():
                                f_pending.cancel()
                        break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Partial outputs were saved.")

    predictions = [predictions_map[k] for k in sorted(predictions_map.keys())]
    metrics = compute_metrics(predictions)
    summary = {
        "gold_jsonl": str(gold_path),
        "verify_model": args.verify_model,
        "workers": args.workers,
        "max_attempts": args.max_attempts,
        "pipeline_preset": args.preset,
        "run_dft": bool(args.dft),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    write_jsonl(predictions_path, predictions)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        write_confusion_csv(outdir / "confusion_matrix.csv", confusion)

    with open(progress_path, "w", encoding="utf-8") as pf:
        json.dump(
            {
                "processed": len(predictions),
                "total": len(rows),
                "status": "interrupted" if interrupted else "completed",
                "timestamp": datetime.now().isoformat(),
            },
            pf,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== PhyVer batch evaluation complete ===")
    print(f"Output dir: {outdir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
