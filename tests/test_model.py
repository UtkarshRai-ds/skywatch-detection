"""Tests for src/models/evaluate.py and src/models/compare.py.

JSON-backed tests (class_stats.json, training_history.json) are skipped when
the files are absent so the suite passes in environments where run_evaluation.py
has not yet been executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.models.compare import compare_models, identify_challenging_classes
from src.models.evaluate import build_test_results, ModelInfo

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_model_info(name: str, params: int) -> ModelInfo:
    return ModelInfo(
        model_name=name,
        path=f"models/{name}_best.pt",
        params=params,
        architecture=name.upper(),
        task="detect",
    )


def _make_result(
    name: str,
    map50: float,
    per_class: dict[str, float],
    params: int = 3_000_000,
) -> dict[str, Any]:
    """Build a result dict as produced by build_test_results."""
    info = _fake_model_info(name, params)
    return build_test_results(
        model_info=info,
        overall={"mAP50": map50, "precision": 0.80, "recall": 0.70, "mAP50_95": 0.40},
        per_class=per_class,
    )


# ── build_test_results ─────────────────────────────────────────────────────────

def test_build_test_results_has_required_keys() -> None:
    result = _make_result("yolov8n", 0.689, {"airplane": 0.561, "ship": 0.782})
    for key in ("model_name", "architecture", "params", "overall", "per_class_mAP50",
                "class_ranking", "best_class", "worst_class"):
        assert key in result


def test_build_test_results_normalises_class_names() -> None:
    result = _make_result("yolov8n", 0.689, {"storage tank": 0.895, "ground track field": 0.579})
    assert "storage_tank" in result["per_class_mAP50"]
    assert "ground_track_field" in result["per_class_mAP50"]
    assert "storage tank" not in result["per_class_mAP50"]


def test_build_test_results_identifies_best_and_worst_class() -> None:
    result = _make_result(
        "yolov8n", 0.689,
        {"airplane": 0.561, "vehicle": 0.952, "bridge": 0.030},
    )
    assert result["best_class"] == "vehicle"
    assert result["worst_class"] == "bridge"


# ── compare_models ─────────────────────────────────────────────────────────────

RESULT_N = _make_result(
    "yolov8n", 0.689,
    {"airplane": 0.561, "ship": 0.782, "bridge": 0.030},
    params=3_000_000,
)
RESULT_S = _make_result(
    "yolov8s", 0.697,
    {"airplane": 0.675, "ship": 0.764, "bridge": 0.027},
    params=11_000_000,
)


def test_compare_models_winner_is_higher_map50() -> None:
    comparison = compare_models(RESULT_N, RESULT_S)
    # yolov8s mAP50=0.697 > yolov8n mAP50=0.689
    assert comparison["winner_mAP50"] == "yolov8s"


def test_compare_models_returns_expected_keys() -> None:
    comparison = compare_models(RESULT_N, RESULT_S)
    for key in ("models", "overall_delta", "per_class_delta",
                "classes_b_wins", "classes_a_wins", "param_ratio",
                "winner_mAP50", "recommendation"):
        assert key in comparison


def test_per_class_delta_airplane_positive() -> None:
    comparison = compare_models(RESULT_N, RESULT_S)
    # airplane: 0.675 (s) − 0.561 (n) = +0.114  →  yolov8s wins
    delta = comparison["per_class_delta"]["airplane"]["delta"]
    assert pytest.approx(delta, abs=1e-3) == 0.114
    assert "airplane" in comparison["classes_b_wins"]


def test_per_class_delta_ship_negative() -> None:
    comparison = compare_models(RESULT_N, RESULT_S)
    # ship: 0.764 (s) − 0.782 (n) = −0.018  →  yolov8n wins
    delta = comparison["per_class_delta"]["ship"]["delta"]
    assert delta < 0
    assert "ship" in comparison["classes_a_wins"]


def test_compare_models_param_ratio() -> None:
    comparison = compare_models(RESULT_N, RESULT_S)
    # 11_000_000 / 3_000_000 ≈ 3.67
    assert comparison["param_ratio"] == pytest.approx(11_000_000 / 3_000_000, rel=1e-2)


def test_identify_challenging_classes_below_threshold() -> None:
    result = identify_challenging_classes([RESULT_N, RESULT_S], threshold=0.5)
    # bridge is below 0.5 for both models (0.030 and 0.027)
    assert "bridge" in result["challenging_classes"]


# ── JSON output files (skipped when absent) ────────────────────────────────────

@pytest.mark.skipif(
    not (DATA_DIR / "class_stats.json").exists(),
    reason="class_stats.json not generated yet — run run_evaluation.py first",
)
def test_class_stats_json_top_level_keys() -> None:
    with (DATA_DIR / "class_stats.json").open(encoding="utf-8") as fh:
        cs = json.load(fh)

    for key in ("models", "comparison", "speed_accuracy_tradeoff",
                "challenging_classes", "prediction_summary"):
        assert key in cs, f"Missing top-level key: {key}"


@pytest.mark.skipif(
    not (DATA_DIR / "class_stats.json").exists(),
    reason="class_stats.json not generated yet — run run_evaluation.py first",
)
def test_class_stats_json_model_structure() -> None:
    with (DATA_DIR / "class_stats.json").open(encoding="utf-8") as fh:
        cs = json.load(fh)

    for model in ("yolov8n", "yolov8s"):
        assert model in cs["models"], f"Missing model: {model}"
        m = cs["models"][model]
        assert "overall" in m
        assert "per_class_mAP50" in m
        for metric in ("mAP50", "precision", "recall", "mAP50_95"):
            assert metric in m["overall"], f"Missing metric {metric} for {model}"


@pytest.mark.skipif(
    not (DATA_DIR / "training_history.json").exists(),
    reason="training_history.json not generated yet — run run_evaluation.py first",
)
def test_training_history_has_50_epochs_for_both_models() -> None:
    with (DATA_DIR / "training_history.json").open(encoding="utf-8") as fh:
        th = json.load(fh)

    for model in ("yolov8n", "yolov8s"):
        assert model in th, f"Missing model in training history: {model}"
        assert th[model]["epochs"] == 50, f"{model}: expected 50 epochs"
        assert len(th[model]["history"]) == 50, f"{model}: history length != 50"


@pytest.mark.skipif(
    not (DATA_DIR / "training_history.json").exists(),
    reason="training_history.json not generated yet — run run_evaluation.py first",
)
def test_training_history_epoch_dict_has_required_fields() -> None:
    with (DATA_DIR / "training_history.json").open(encoding="utf-8") as fh:
        th = json.load(fh)

    required = {"epoch", "train_box_loss", "val_box_loss", "mAP50", "mAP50_95"}
    for model in ("yolov8n", "yolov8s"):
        first_epoch = th[model]["history"][0]
        for field in required:
            assert field in first_epoch, f"{model} epoch[0] missing field: {field}"
