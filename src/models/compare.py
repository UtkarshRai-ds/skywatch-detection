"""Side-by-side comparison of two trained YOLOv8 models.

Computes per-class AP deltas, speed vs accuracy trade-off metrics, and
generates a structured comparison dict for downstream reporting and plotting.

Typical usage::

    from src.models.compare import compare_models, speed_accuracy_tradeoff

    comparison = compare_models(results_n, results_s)
    tradeoff   = speed_accuracy_tradeoff([results_n, results_s])
"""

from __future__ import annotations

from typing import Any

from src.data.loader import NWPU_CLASSES


def compare_models(
    results_a: dict[str, Any],
    results_b: dict[str, Any],
) -> dict[str, Any]:
    """Compute a comprehensive side-by-side comparison of two model result dicts.

    Both result dicts are expected to have been produced by
    :func:`src.models.evaluate.build_test_results`.

    Positive delta values mean model B outperforms model A.

    Args:
        results_a: Evaluation results for the first model (baseline).
        results_b: Evaluation results for the second model (comparison).

    Returns:
        Dict with keys:

        - ``models``: names of the two compared models.
        - ``overall_delta``: {metric: (a_val, b_val, delta)} for each metric.
        - ``per_class_delta``: {class_name: (a_mAP50, b_mAP50, delta)}.
        - ``classes_b_wins``: list of class names where B has higher mAP50.
        - ``classes_a_wins``: list of class names where A has higher mAP50.
        - ``param_ratio``: B.params / A.params — relative model size.
        - ``mAP50_per_million_params``: efficiency metric for both models.
        - ``winner_mAP50``: name of the model with higher mAP50.
        - ``recommendation``: one-sentence deployment guidance.
    """
    name_a = results_a["model_name"]
    name_b = results_b["model_name"]

    ov_a = results_a["overall"]
    ov_b = results_b["overall"]

    # Overall metric deltas
    overall_delta: dict[str, dict[str, float]] = {}
    for metric in ("mAP50", "precision", "recall", "mAP50_95"):
        va = ov_a[metric]
        vb = ov_b[metric]
        overall_delta[metric] = {
            name_a: va,
            name_b: vb,
            "delta": round(vb - va, 4),
            "delta_pct": round((vb - va) / max(va, 1e-9) * 100, 2),
        }

    # Per-class mAP50 deltas
    pc_a = results_a["per_class_mAP50"]
    pc_b = results_b["per_class_mAP50"]
    all_classes = sorted(set(pc_a) | set(pc_b))

    per_class_delta: dict[str, dict[str, float]] = {}
    classes_b_wins: list[str] = []
    classes_a_wins: list[str] = []

    for cls in all_classes:
        va = pc_a.get(cls, 0.0)
        vb = pc_b.get(cls, 0.0)
        delta = round(vb - va, 4)
        per_class_delta[cls] = {name_a: va, name_b: vb, "delta": delta}
        if delta > 0:
            classes_b_wins.append(cls)
        elif delta < 0:
            classes_a_wins.append(cls)

    # Sort by absolute delta magnitude for impact ranking
    sorted_by_impact = sorted(
        per_class_delta.items(),
        key=lambda x: abs(x[1]["delta"]),
        reverse=True,
    )

    # Efficiency: mAP50 per million parameters
    params_a_m = results_a["params"] / 1_000_000
    params_b_m = results_b["params"] / 1_000_000
    eff_a = round(ov_a["mAP50"] / params_a_m, 4) if params_a_m > 0 else 0.0
    eff_b = round(ov_b["mAP50"] / params_b_m, 4) if params_b_m > 0 else 0.0

    winner = name_b if ov_b["mAP50"] >= ov_a["mAP50"] else name_a
    loser  = name_a if winner == name_b else name_b

    map_delta = overall_delta["mAP50"]["delta"]
    param_ratio = round(results_b["params"] / max(results_a["params"], 1), 2)

    recommendation = (
        f"Deploy {winner} for maximum accuracy "
        f"(mAP50 {overall_delta['mAP50'][winner]:.3f}, "
        f"+{abs(map_delta):.3f} over {loser}). "
        f"{name_a} offers {param_ratio:.1f}× fewer parameters "
        f"with {eff_a:.3f} mAP50/M-params vs {eff_b:.3f} for {name_b} — "
        f"prefer {name_a} for edge or latency-constrained deployments."
        if winner == name_b
        else
        f"{name_a} leads with mAP50 {ov_a['mAP50']:.3f} despite fewer "
        f"parameters ({params_a_m:.1f}M vs {params_b_m:.1f}M); "
        f"{name_b}'s size increase does not justify the accuracy gain here."
    )

    return {
        "models": [name_a, name_b],
        "overall_delta": overall_delta,
        "per_class_delta": per_class_delta,
        "sorted_by_impact": [k for k, _ in sorted_by_impact],
        "classes_b_wins": sorted(classes_b_wins),
        "classes_a_wins": sorted(classes_a_wins),
        "param_ratio": param_ratio,
        "mAP50_per_million_params": {name_a: eff_a, name_b: eff_b},
        "winner_mAP50": winner,
        "recommendation": recommendation,
    }


def speed_accuracy_tradeoff(
    results_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a speed vs accuracy table for plotting a trade-off scatter.

    Args:
        results_list: List of result dicts from
            :func:`src.models.evaluate.build_test_results`.

    Returns:
        List of dicts, one per model, with keys ``model_name``, ``mAP50``,
        ``mAP50_95``, ``params_M``, ``total_ms``, ``fps``, ``params_label``.
    """
    points: list[dict[str, Any]] = []
    for r in results_list:
        spd = r.get("speed", {})
        params_m = round(r["params"] / 1_000_000, 2)
        points.append(
            {
                "model_name":  r["model_name"],
                "mAP50":       r["overall"]["mAP50"],
                "mAP50_95":    r["overall"]["mAP50_95"],
                "params_M":    params_m,
                "params_label": f"{params_m:.1f}M",
                "total_ms":    spd.get("total_ms", None),
                "fps":         spd.get("fps", None),
                "device":      spd.get("device", "unknown"),
            }
        )
    return points


def identify_challenging_classes(
    results_list: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Find classes that consistently score below a mAP50 threshold.

    Args:
        results_list: List of model result dicts.
        threshold: mAP50 below which a class is considered challenging.

    Returns:
        Dict with ``challenging_classes`` (appear below threshold in ALL
        models), ``per_model_low`` (per-model lists), and ``analysis`` text.
    """
    per_model_low: dict[str, list[str]] = {}

    for r in results_list:
        low = [
            cls for cls, ap in r["per_class_mAP50"].items()
            if ap < threshold
        ]
        per_model_low[r["model_name"]] = sorted(low)

    # Intersection — classes that are hard for every model
    if per_model_low:
        challenging = sorted(
            set.intersection(*[set(v) for v in per_model_low.values()])
        )
    else:
        challenging = []

    analysis_parts: list[str] = []
    for cls in challenging:
        vals = {r["model_name"]: r["per_class_mAP50"].get(cls, 0.0) for r in results_list}
        val_str = ", ".join(f"{k}={v:.3f}" for k, v in vals.items())
        analysis_parts.append(f"{cls}: {val_str}")

    return {
        "threshold": threshold,
        "challenging_classes": challenging,
        "per_model_low": per_model_low,
        "analysis": " | ".join(analysis_parts) if analysis_parts else "No classes below threshold.",
    }
