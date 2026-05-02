#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claim_verification_llm import _call_llm_json


LIKERT_LEVELS = [-2, -1, 0, 1, 2]

SYSTEM_PROMPT = """
You are an expert materials-science claim verifier.
Given a claim and metadata, produce a single Likert truthfulness score using exactly one of: -2, -1, 0, 1, 2.

Interpretation:
-2: strongly not correct
-1: somewhat not correct
 0: uncertain / mixed
 1: mostly correct
 2: strongly correct

Return ONLY valid JSON:
{
  "score": -2,
  "reason": "short explanation"
}
""".strip()


def read_gold_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "claim" not in obj or "likert_score" not in obj:
                continue
            records.append(
                {
                    "line_no": line_no,
                    "problem_id": obj.get("problem_id"),
                    "claim": str(obj.get("claim", "")).strip(),
                    "gold_score": int(obj.get("likert_score")),
                    "domain": obj.get("domain"),
                    "subdomain": obj.get("subdomain"),
                }
            )
    return records


def _extract_score_from_raw_text(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    candidates = re.findall(r"(?<!\d)(-2|-1|0|1|2)(?!\d)", raw)
    if not candidates:
        return None
    return int(candidates[0])


def query_likert_score(
    claim_row: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    retries: int,
    retry_sleep_s: float,
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str]]:
    payload = {
        "problem_id": claim_row.get("problem_id"),
        "domain": claim_row.get("domain"),
        "subdomain": claim_row.get("subdomain"),
        "claim": claim_row.get("claim"),
        "task": "Assign Likert score only.",
    }
    user_prompt = json.dumps(payload, ensure_ascii=False)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        parsed, raw_text, model_used = _call_llm_json(
            designer_client=model,
            api_key=api_key,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        score: Optional[int] = None
        reason: Optional[str] = None

        if isinstance(parsed, dict):
            raw_score = parsed.get("score")
            if isinstance(raw_score, (int, float)) and int(round(float(raw_score))) in LIKERT_LEVELS:
                score = int(round(float(raw_score)))
            reason = str(parsed.get("reason", "")).strip() or None

        if score is None:
            score = _extract_score_from_raw_text(raw_text if isinstance(raw_text, str) else None)

        if score in LIKERT_LEVELS:
            return score, reason, (raw_text if isinstance(raw_text, str) else None), model_used

        last_err = f"Invalid or missing score from model output (attempt {attempt + 1})."
        if attempt < retries:
            time.sleep(retry_sleep_s)

    return None, None, None, last_err


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
    supports: Dict[int, int] = defaultdict(int)
    f1_vals: List[float] = []
    for cls in LIKERT_LEVELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        supports[cls] = sum(1 for g in gold if g == cls)

        prec = (tp / (tp + fp)) if (tp + fp) else None
        rec = (tp / (tp + fn)) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
        if f1 is not None:
            f1_vals.append(f1)
        per_class[cls] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": supports[cls],
        }

    metrics["per_class"] = per_class
    metrics["macro_f1"] = _safe_mean(f1_vals)
    return metrics


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_error_log(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    problem_id = row.get("problem_id")
    status = row.get("status")
    err = row.get("error")
    raw = row.get("llm_raw")
    raw_preview = None
    if isinstance(raw, str):
        raw_preview = raw[:500].replace("\n", "\\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] problem_id={problem_id} status={status} error={err}\n")
        if raw_preview:
            f.write(f"  llm_raw_preview={raw_preview}\n")


def write_confusion_csv(path: Path, confusion: Dict[int, Dict[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("gold\\pred," + ",".join(str(x) for x in LIKERT_LEVELS) + "\n")
        for g in LIKERT_LEVELS:
            row = [str(confusion.get(g, {}).get(p, 0)) for p in LIKERT_LEVELS]
            f.write(f"{g}," + ",".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LLM-only Likert claim verification and compare with gold labels.")
    ap.add_argument("--gold-jsonl", required=True, help="Path to a gold JSONL evaluation file.")
    ap.add_argument("--outdir", default=None, help="Output directory. Default: ./artifacts/batch_runs/llm_eval_<timestamp>")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry-sleep", type=float, default=1.5)
    ap.add_argument("--dry-run", action="store_true", help="No LLM calls; only validate input parsing and output scaffolding.")
    args = ap.parse_args()

    gold_path = Path(args.gold_jsonl).resolve()
    rows = read_gold_jsonl(gold_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No valid claim rows found in: {gold_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir).resolve() if args.outdir else (Path("./artifacts/batch_runs") / f"llm_eval_{stamp}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = outdir / "predictions.jsonl"
    errors_path = outdir / "errors.log"
    progress_path = outdir / "progress.json"

    predictions_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")

    api_key = args.api_key
    if not api_key:
        api_key = None

    predictions: List[Dict[str, Any]] = []
    interrupted = False
    try:
        for i, row in enumerate(rows, 1):
            rec = dict(row)

            if args.dry_run:
                rec.update(
                    {
                        "pred_score": None,
                        "pred_reason": None,
                        "model_used": args.model,
                        "status": "dry_run",
                    }
                )
                predictions.append(rec)
                append_jsonl_row(predictions_path, rec)
                with open(progress_path, "w", encoding="utf-8") as pf:
                    json.dump({"processed": i, "total": len(rows), "status": "running", "last_problem_id": row.get("problem_id")}, pf, ensure_ascii=False, indent=2)
                continue

            score, reason, raw_text, model_used_or_err = query_likert_score(
                claim_row=row,
                model=args.model,
                api_key=api_key,
                retries=max(0, args.retries),
                retry_sleep_s=max(0.0, args.retry_sleep),
            )
            if score is None:
                rec.update(
                    {
                        "pred_score": None,
                        "pred_reason": None,
                        "model_used": args.model,
                        "status": "error",
                        "error": model_used_or_err,
                        "llm_raw": raw_text,
                    }
                )
                append_error_log(errors_path, rec)
            else:
                rec.update(
                    {
                        "pred_score": int(score),
                        "pred_reason": reason,
                        "model_used": model_used_or_err,
                        "status": "ok",
                        "llm_raw": raw_text,
                        "abs_error": abs(int(score) - int(row["gold_score"])),
                    }
                )

            predictions.append(rec)
            append_jsonl_row(predictions_path, rec)
            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({"processed": i, "total": len(rows), "status": "running", "last_problem_id": row.get("problem_id")}, pf, ensure_ascii=False, indent=2)

            print(f"[{i:02d}/{len(rows)}] {row.get('problem_id')} gold={row.get('gold_score')} pred={rec.get('pred_score')} status={rec.get('status')}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Partial outputs were saved.")

    metrics = compute_metrics(predictions)
    run_summary = {
        "gold_jsonl": str(gold_path),
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    write_jsonl(predictions_path, predictions)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

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

    confusion = metrics.get("confusion_matrix")
    if isinstance(confusion, dict):
        write_confusion_csv(outdir / "confusion_matrix.csv", confusion)

    print("\n=== Evaluation complete ===")
    print(f"Output dir: {outdir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
