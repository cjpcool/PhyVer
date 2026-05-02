from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = """
You are an expert materials scientist and verification reviewer.
Your task is to verify whether the claim is correct or the claimed material is feasible,
by considering both the claim itself and the additional support materials:
1) generated material structure summary, and
2) DFT results.

You MUST:
- Ground judgment in provided data only.
- Compare key quantitative parameters (e.g., gap_eV, dipole_D, HOMO/LUMO, energy trends, notable force magnitudes if relevant).
- Return a calibrated Likert score using exactly one of: -2, -1, 0, 1, 2.
    -2: strongly not correct / clearly infeasible
    -1: somewhat not correct / likely infeasible
     0: uncertain or mixed evidence
     1: mostly correct / likely feasible
     2: strongly correct / clearly feasible
- Be conservative when data is missing.

Return ONLY valid JSON with this schema:
{
  "verdict": "supported|partially-supported|not-supported|insufficient-evidence",
    "score": -2,
  "reason": "short explanation in plain language",
  "parameter_comparisons": [
    {
      "property": "gap_eV",
      "status": "pass|fail|uncertain",
      "claim_target": "what claim expects",
      "actual": "value from DFT/structure",
      "comparison": "why pass/fail",
      "importance": "high|medium|low"
    }
  ],
  "key_constraints": [
    {
      "property": "gap_eV",
      "target": "semiconductor around 0.3 eV",
      "source": "parsed from claim"
    }
  ]
}
""".strip()

LIKERT_LEVELS = (-2, -1, 0, 1, 2)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _structure_summary(structure: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not structure:
        return {"available": False}

    atoms = structure.get("atoms") or []
    symbols = [a.get("symbol") for a in atoms if isinstance(a, dict) and a.get("symbol")]
    comp = Counter(symbols)

    frac = None
    if atoms and isinstance(atoms, list):
        frac = atoms[: min(8, len(atoms))]

    return {
        "available": True,
        "source": structure.get("source"),
        "format": structure.get("format"),
        "n_atoms": structure.get("n_atoms"),
        "lengths": structure.get("lengths"),
        "angles": structure.get("angles"),
        "cell": structure.get("cell"),
        "composition": dict(comp),
        "atom_preview": frac,
    }


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.S | re.I)
    cand = fence.group(1).strip() if fence else stripped

    try:
        return json.loads(cand)
    except Exception:
        pass

    first = cand.find("{")
    last = cand.rfind("}")
    if first >= 0 and last > first:
        maybe = cand[first : last + 1]
        try:
            return json.loads(maybe)
        except Exception:
            return None
    return None


def _call_llm_json(
    *,
    designer_client: str,
    api_key: Optional[str],
    system_prompt: str,
    user_prompt: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    model = (designer_client or "").strip()
    if not model:
        model = "gpt-5"

    lower = model.lower()

    try:
        if "gemini" in lower:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=user_prompt,
            )
            text = getattr(response, "text", "") or ""
            parsed = _extract_json_blob(text)
            return parsed, text, model

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
        )
        text = getattr(response, "output_text", "") or ""
        parsed = _extract_json_blob(text)
        return parsed, text, model
    except Exception as e:
        return None, str(e), model


def _heuristic_verify(claim: str, dft: Dict[str, Any]) -> Dict[str, Any]:
    txt = (claim or "").lower()
    checks: List[Dict[str, Any]] = []

    gap = _safe_float(dft.get("gap_eV"))
    dip = _safe_float(dft.get("dipole_D"))

    if "metal" in txt or "metallic" in txt:
        ok = gap is not None and gap <= 0.10
        checks.append({"property": "gap_eV", "status": "pass" if ok else "fail", "claim_target": "<= 0.10", "actual": gap, "comparison": "metal should have near-zero gap", "importance": "high"})
    if "semiconductor" in txt:
        ok = gap is not None and 0.10 <= gap <= 3.00
        checks.append({"property": "gap_eV", "status": "pass" if ok else "fail", "claim_target": "0.10 to 3.00", "actual": gap, "comparison": "typical semiconductor gap range", "importance": "high"})
    if "insulator" in txt:
        ok = gap is not None and gap >= 3.00
        checks.append({"property": "gap_eV", "status": "pass" if ok else "fail", "claim_target": ">= 3.00", "actual": gap, "comparison": "insulator generally has large gap", "importance": "high"})

    m = re.search(r"(band\s*gap|gap)\s*(?:of|=|:|around|~)?\s*([0-9]+(?:\.[0-9]+)?)\s*e?v", txt)
    if m and gap is not None:
        target = float(m.group(2))
        ok = abs(gap - target) <= 0.2
        checks.append({"property": "gap_eV", "status": "pass" if ok else "fail", "claim_target": f"~{target}±0.2", "actual": gap, "comparison": "explicit claimed band gap", "importance": "high"})

    if "high dipole" in txt:
        ok = dip is not None and dip >= 2.0
        checks.append({"property": "dipole_D", "status": "pass" if ok else "fail", "claim_target": ">= 2.0", "actual": dip, "comparison": "high dipole requirement", "importance": "medium"})
    if "low dipole" in txt or "small dipole" in txt:
        ok = dip is not None and dip <= 0.5
        checks.append({"property": "dipole_D", "status": "pass" if ok else "fail", "claim_target": "<= 0.5", "actual": dip, "comparison": "low dipole requirement", "importance": "medium"})

    known = [c for c in checks if c.get("status") in {"pass", "fail"}]
    pass_n = sum(1 for c in known if c["status"] == "pass")
    fail_n = sum(1 for c in known if c["status"] == "fail")
    ratio = float(pass_n / len(known)) if known else 0.0

    if ratio >= 0.90:
        score = 2
    elif ratio >= 0.65:
        score = 1
    elif ratio >= 0.35:
        score = 0
    elif ratio >= 0.10:
        score = -1
    else:
        score = -2

    if not known:
        verdict = "insufficient-evidence"
        reason = "Claim lacks mappable quantitative constraints for provided DFT fields."
    elif fail_n == 0:
        verdict = "supported"
        reason = "All mapped quantitative checks pass."
    elif pass_n == 0:
        verdict = "not-supported"
        reason = "Mapped quantitative checks fail."
    else:
        verdict = "partially-supported"
        reason = "Some checks pass while others fail."

    return {
        "verdict": verdict,
        "score": score,
        "reason": reason,
        "parameter_comparisons": checks,
        "key_constraints": [],
        "_fallback": True,
    }


def verify_claim_with_llm(
    *,
    claim: str,
    generated_structure: Optional[Dict[str, Any]],
    dft: Dict[str, Any],
    designer_client: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    structure_summary = _structure_summary(generated_structure)

    user_payload = {
        "claim": claim,
        "generated_structure_summary": structure_summary,
        "dft_results": dft,
        "instructions": {
            "focus": [
                "Compare claim targets against DFT values",
                "Use structure information when relevant to interpretation",
                "Provide concise reason and key parameter comparisons",
            ]
        },
    }

    user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)

    parsed, raw_text, model_used = _call_llm_json(
        designer_client=designer_client or "gpt-5",
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    if not isinstance(parsed, dict):
        parsed = _heuristic_verify(claim, dft)
        parsed["reason"] = f"{parsed.get('reason', '')} (LLM call unavailable or invalid JSON: {raw_text})".strip()

    verdict = str(parsed.get("verdict", "insufficient-evidence")).strip() or "insufficient-evidence"
    if verdict not in {"supported", "partially-supported", "not-supported", "insufficient-evidence"}:
        verdict = "insufficient-evidence"

    raw_score = parsed.get("score")
    score_f = _safe_float(raw_score)
    if score_f is None:
        score = 0
    else:
        score = int(round(score_f))
        score = min(max(score, -2), 2)
    if score not in LIKERT_LEVELS:
        score = 0

    reason = str(parsed.get("reason", "")).strip()
    comparisons = parsed.get("parameter_comparisons")
    if not isinstance(comparisons, list):
        comparisons = []

    constraints = parsed.get("key_constraints")
    if not isinstance(constraints, list):
        constraints = []

    normalized_checks: List[Dict[str, Any]] = []
    for c in comparisons:
        if not isinstance(c, dict):
            continue
        status = str(c.get("status", "uncertain")).strip().lower()
        if status not in {"pass", "fail", "uncertain", "unknown"}:
            status = "uncertain"
        normalized_checks.append(
            {
                "property": c.get("property") or "unknown",
                "status": status,
                "target": c.get("claim_target"),
                "actual": c.get("actual"),
                "op": c.get("comparison"),
                "source": "llm-judge",
                "importance": c.get("importance"),
                "reason": c.get("comparison"),
            }
        )

    return {
        "verdict": verdict,
        "score": score,
        "reason": reason,
        "checks": normalized_checks,
        "extracted_constraints": constraints,
        "parameter_comparisons": comparisons,
        "dft_used": dft,
        "model_used": model_used,
        "llm_raw": raw_text if isinstance(raw_text, str) else None,
    }
