"""Models page: the trained networks side by side, and which one the app serves."""

from __future__ import annotations

import streamlit as st

from . import charts, reports, style

METRICS = {
    "Test accuracy": ("test_accuracy", "Test accuracy", ".2%"),
    "Test errors": ("test_errors", "Wrong answers on the test set", ".0f"),
    "Validation log-loss": ("val_log_loss", "Validation log-loss (lower is better)", ".4f"),
    "Parameters": ("params_millions", "Parameters (millions)", ".1f"),
    "CPU time": ("latency_ms_cpu", "CPU time per photo on an M2 (ms)", ".0f"),
    "Training time": ("train_minutes", "Training time (minutes)", ".1f"),
}
CURVES = {
    "Validation macro-F1": ("val_macro_f1", "Validation macro-F1"),
    "Validation loss": ("val_loss", "Validation loss"),
    "Training loss": ("train_loss", "Training loss"),
    "Training accuracy": ("train_acc", "Training accuracy"),
}


def render() -> None:
    df = reports.comparison()
    style.intro("Four models, one test set.",
                "MobileNetV2, ResNet50 and EfficientNet-B0 were fine-tuned from ImageNet. A fourth model keeps "
                "MobileNetV2's ImageNet layers frozen and trains only the final classifier. All four were trained "
                "on the same leaf-grouped split and tested on the same held-out photos.")
    if df is None:
        st.info("The model comparison report is not available in this deployment.")
        return
    chosen = df.loc[df["selected"]].iloc[0] if df["selected"].any() else df.iloc[0]

    style.figures([
        (f"{len(df)}", "models trained on an 8 GB Apple M2"),
        (f"{chosen['test_accuracy']:.1%}", f"test accuracy of the served model, {int(chosen['test_errors'])} errors "
                                           f"in {int(chosen['test_n']):,} photos"),
        (chosen["label"], f"served in this app: {chosen['params_millions']:.1f} M parameters, "
                          f"{chosen['train_minutes']:.1f} min of training"),
    ])

    style.section("Which model the app serves")
    others = df[~df["selected"]]
    lo, hi = others["test_accuracy"].min(), others["test_accuracy"].max()
    span = f"{lo:.1%}" if lo == hi else f"{lo:.1%} to {hi:.1%}"
    style.html(f'<div class="pick">The app serves <b>{chosen["label"]}</b>: only its last layer was trained, so it '
               f"is the simplest and fastest to train of the four ({chosen['train_minutes']:.1f} min). The fine-tuned "
               f"networks scored higher on this test set ({span}, at most {int(others['test_errors'].max())} errors), "
               "but lab photos from one dataset are an easy test; the simpler model's errors make its limits "
               "easier to see.</div>")

    style.section("Compare the models", "Pick a measure. The served model is highlighted.")
    choice = st.segmented_control("Measure", list(METRICS), default="Test accuracy", label_visibility="collapsed")
    column, title, fmt = METRICS[choice or "Test accuracy"]
    st.altair_chart(charts.metric_bars(df, column, title, fmt, chosen["model"]), width="stretch", theme=None)

    hist = reports.histories()
    if hist is not None:
        style.section("Training, epoch by epoch",
                      "Circles are the head-only phase, diamonds the fine-tuning phase. Click a legend entry to focus on one model.")
        curve = st.segmented_control("Curve", list(CURVES), default="Validation loss", label_visibility="collapsed")
        column, title = CURVES[curve or "Validation loss"]
        st.altair_chart(charts.training_curves(hist, column, title, chosen["label"]), width="stretch", theme=None)

    style.section("All measurements")
    table = df[["label", "test_accuracy", "test_errors", "test_macro_f1", "params_millions", "checkpoint_mb",
                "latency_ms_cpu", "train_minutes", "epochs_run", "val_log_loss"]]
    st.dataframe(table, hide_index=True, column_config={
        "label": "Model",
        "test_accuracy": st.column_config.NumberColumn("Test accuracy", format="percent"),
        "test_errors": st.column_config.NumberColumn("Test errors", format="%d"),
        "test_macro_f1": st.column_config.NumberColumn("Test macro-F1", format="%.4f"),
        "params_millions": st.column_config.NumberColumn("Parameters (M)", format="%.1f"),
        "checkpoint_mb": st.column_config.NumberColumn("File (MB)", format="%.1f"),
        "latency_ms_cpu": st.column_config.NumberColumn("CPU ms / photo", format="%.0f"),
        "train_minutes": st.column_config.NumberColumn("Training (min)", format="%.1f"),
        "epochs_run": st.column_config.NumberColumn("Epochs"),
        "val_log_loss": st.column_config.NumberColumn("Val log-loss", format="%.4f"),
    })
    st.caption(f"Test set: {int(chosen['test_n']):,} held-out photos. Timings are batch size 1 on the training "
               "machine (Apple M2).")
