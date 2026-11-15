"""Serve the frozen-backbone MobileNetV2 baseline in the app instead of the compare.py winner.

The baseline trains only the classifier head on top of the frozen ImageNet backbone
(`python -m grapevine.train --model mobilenet_v2 --head-epochs 2 --finetune-epochs 0
--run-name mobilenet_v2_frozen`). It is lighter and a more modest model than the fine-tuned
networks; its test figures are reported as measured.

    python -m grapevine.serve_baseline
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pandas as pd

from .compare import model_row
from .config import BEST_MODEL_DIR, DATA_REPORT_JSON, MODEL_NAMES, REPORTS_DIR, RUNS_DIR
from .utils import load_json, project_relative, save_json

BASELINE = "mobilenet_v2_frozen"
SELECTION_RULE = ("deployment choice: frozen-backbone MobileNetV2 (classifier head only, no fine-tuning), "
                  "the simplest of the trained models")


def main() -> None:
    rows = [r for r in (model_row(n) for n in (*MODEL_NAMES, BASELINE)) if r is not None]
    table = pd.DataFrame(rows)
    table["selected"] = table["model"] == BASELINE
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    shutil.copy2(RUNS_DIR / BASELINE / "best.pt", BEST_MODEL_DIR / "model.pt")
    metrics = load_json(REPORTS_DIR / BASELINE / "metrics.json")
    data_report = load_json(DATA_REPORT_JSON) if DATA_REPORT_JSON.exists() else {}
    card = {
        "model": BASELINE,
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": SELECTION_RULE,
        "checkpoint": project_relative(BEST_MODEL_DIR / "model.pt"),
        "class_names": list(metrics["test"]["all"]["per_class"].keys()),
        "input": "RGB, resized to 224x224, ImageNet mean/std normalisation",
        "validation": metrics["val"],
        "test": metrics["test"],
        "efficiency": metrics["efficiency"],
        "test_confidence": metrics.get("test_confidence"),
        "background_neutralized_test": metrics.get("background_neutralized_test"),
        "test_accuracy_exact_95ci_one_image_per_leaf": metrics.get("test_accuracy_exact_95ci_one_image_per_leaf"),
        "training": load_json(RUNS_DIR / BASELINE / "train_summary.json"),
        "data": {"splits": data_report.get("splits"), "leakage_checks": data_report.get("leakage_checks")},
        "intended_use": "Lab-style photos of single grape leaves on a plain background (PlantVillage-like). "
                        "Not validated on field images, other crops or other diseases.",
    }
    save_json(card, BEST_MODEL_DIR / "model_card.json")
    t = metrics["test"]["all"]
    print(f"Serving {BASELINE}: test accuracy {t['accuracy']:.4f}, {t['errors']} errors in {t['n']} images")


if __name__ == "__main__":
    main()
