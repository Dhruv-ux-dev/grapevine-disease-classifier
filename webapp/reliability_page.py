"""Reliability page: test accuracy and errors, shortcut checks, confidence and known limits."""

from __future__ import annotations

import streamlit as st

from . import charts, reports, style
from .diagnose import get_predictor


def render() -> None:
    model = reports.served_name(get_predictor().model_name)
    summary = reports.summary(model)
    bias = reports.background_bias() or {}
    card = reports.model_card() or {}
    style.intro("How far to trust it.",
                "Accuracy on held-out photos, where the mistakes are, and whether the model is looking at the leaf "
                "or at the background of the photo.")
    if not summary or summary.get("test_accuracy") is None:
        st.info("The evaluation report for the served model is not available in this deployment.")
        return

    items = [(f"{summary['test_accuracy']:.1%}",
              f"test accuracy: {summary['test_errors']} wrong answers in {summary['test_images']:,} held-out photos")]
    if summary.get("leaves_correct") is not None:
        items.append((f"{summary['leaves_correct']}/{summary['leaves']}",
                      f"unseen leaves classified correctly, one photo per leaf (95% interval "
                      f"{summary['accuracy_low']:.1%} or better)"))
    if summary.get("background_accuracy"):
        items.append((f"{summary['background_accuracy']:.1%}",
                      f"with the photo background painted grey ({summary['background_errors']} errors in "
                      f"{summary['background_n']:,} photos)"))
    style.figures(items)

    if summary.get("confusion_matrix") and summary.get("class_names"):
        style.section("Where it goes wrong",
                      "Rows are the true class, columns the model's answer. Green cells are correct; red cells are "
                      "mistakes. Hover a cell for its share of the class.")
        st.altair_chart(charts.confusion_matrix(summary["confusion_matrix"], summary["class_names"]),
                        width="stretch", theme=None)

    background_only = (bias.get("test_all") or {}).get("accuracy")
    rows = [{"test": "Chance, 4 classes", "accuracy": bias.get("chance_accuracy", .25), "kind": "Baseline",
             "detail": "Always guessing one class"}]
    if background_only:
        rows.append({"test": "Background pixels only", "accuracy": background_only, "kind": "Baseline",
                     "detail": "Simple classifier on the colour of the photo border"})
    if summary.get("background_accuracy"):
        rows.append({"test": "Model, background painted grey", "accuracy": summary["background_accuracy"],
                     "kind": "Model", "detail": f"{summary['background_n']:,} test photos"})
    rows.append({"test": "Model, normal test photos", "accuracy": summary["test_accuracy"], "kind": "Model",
                 "detail": f"{summary['test_errors']} errors in {summary['test_images']:,} photos"})
    for r in rows:
        r["label"] = f"{r['accuracy']:.1%}"
    drop = summary["test_accuracy"] - (summary.get("background_accuracy") or summary["test_accuracy"])
    verdict = ("It barely moved, so the model relies on the leaf." if drop < 0.02 else
               f"It fell by {drop * 100:.1f} percentage points, so part of the model's accuracy comes from the background rather than the leaf.")
    style.section("Is it looking at the leaf?",
                  f"Background colour alone predicts the class {background_only:.0%} of the time (chance is 25%). "
                  f"Hiding the background tests how much the model uses that shortcut. {verdict}"
                  if background_only else verdict)
    st.altair_chart(charts.robustness(rows), width="stretch", theme=None)

    preds = reports.test_predictions(model)
    if preds is not None:
        conf = card.get("test_confidence") or {}
        wrong = conf.get("median_wrong")
        wrong_text = f" Wrong answers had a median confidence of {wrong:.0%}." if wrong else ""
        style.section("How sure it is",
                      f"Top-class confidence on the {len(preds):,} test photos. The median is "
                      f"{conf.get('median', preds['confidence'].median()):.0%} and 5% of photos fall below "
                      f"{conf.get('p05', preds['confidence'].quantile(.05)):.0%}.{wrong_text} The app warns below 70%.")
        st.altair_chart(charts.confidence_histogram(preds), width="stretch", theme=None)

    style.section("Known limits")
    st.markdown(
        "- **Lab photos only.** Every training and test photo shows one leaf on a plain background. Field photos "
        "with soil, sky or other leaves have not been tested, and accuracy there is likely lower.\n"
        "- **Four classes only.** Other grape problems (downy or powdery mildew, nutrient deficiency, insect damage) "
        "will be forced into one of the four labels.\n"
        "- **Black Rot and ESCA are the most confused pair**, as the matrix above shows.\n"
        "- **One dataset.** Test photos come from the same collection and capture setup as the training photos.\n"
        "- **Single training run.** Each model was trained once; run-to-run variation was not measured."
    )
