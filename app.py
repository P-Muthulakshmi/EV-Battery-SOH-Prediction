"""Flask application for EV Battery State of Health prediction.

This module wires the trained machine learning model into a clean web
application. The model itself is not retrained here; it is loaded from
``models/best_model.pkl`` and used only for inference.
"""
from __future__ import annotations
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


import csv
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "cleaned_battery_dataset.csv"
HISTORY_PATH = BASE_DIR / "reports" / "prediction_history.csv"

FEATURE_COLUMNS = [
    "BatchID",
    "Cycle",
    "Voltage",
    "Current",
    "Temperature",
    "ChargeTime",
    "DischargeTime",
    "InternalResistance",
    "Capacity",
    "AmbientHumidity",
    "C_Rate",
]

FIELD_RULES = {
    "BatchID": {"type": int, "min": 0, "max": 3, "step": 1, "label": "Battery Batch"},
    "Cycle": {"type": int, "min": 1, "max": 2000, "step": 1, "label": "Cycle Count"},
    "Voltage": {"type": float, "min": 3.0, "max": 4.2, "step": 0.01, "label": "Voltage"},
    "Current": {"type": float, "min": 0.5, "max": 2.0, "step": 0.01, "label": "Current"},
    "Temperature": {"type": float, "min": 10, "max": 41, "step": 0.1, "label": "Temperature"},
    "ChargeTime": {
    "type": float,
    "min": 30,
    "max": 120,
    "step": 0.01,
    "label": "Charge Time",
},

"DischargeTime": {
    "type": float,
    "min": 30,
    "max": 120,
    "step": 0.01,
    "label": "Discharge Time",
},
    "InternalResistance": {"type": float, "min": 0.04, "max": 0.06, "step": 0.001, "label": "Internal Resistance"},
    "Capacity": {"type": float, "min": 1.46, "max": 2.52, "step": 0.01, "label": "Capacity"},
    "AmbientHumidity": {"type": float, "min": 30, "max": 70, "step": 0.1, "label": "Ambient Humidity"},
    "C_Rate": {"type": float, "min": 0.5, "max": 2.0, "step": 0.01, "label": "C-Rate"},
}


def create_app() -> Flask:
    """Create and configure the Flask app instance."""
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["SECRET_KEY"] = "change-this-secret-key-before-deployment"

    register_routes(app)
    return app


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the trained Random Forest model once per app process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_battery_condition(soh: float) -> str:
    """Convert a predicted SOH percentage into a readable health label."""
    if soh >= 90:
        return "Healthy"
    if soh >= 80:
        return "Good"
    if soh >= 70:
        return "Moderate"
    return "Replace Soon"


def parse_prediction_form(form_data: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize prediction form values."""
    values: dict[str, Any] = {}
    errors: list[str] = []

    for field in FEATURE_COLUMNS:
        rule = FIELD_RULES[field]
        raw_value = form_data.get(field, "").strip()

        if not raw_value:
            errors.append(f"{rule['label']} is required.")
            continue

        try:
            value = rule["type"](raw_value)
        except ValueError:
            errors.append(f"{rule['label']} must be a valid number.")
            continue

        minimum = rule.get("min")
        maximum = rule.get("max")

        if minimum is not None and value < minimum:
            errors.append(f"{rule['label']} must be at least {minimum}.")
        elif maximum is not None and value > maximum:
            errors.append(f"{rule['label']} must be at most {maximum}.")
        else:
            values[field] = value

    return values, errors


def predict_soh(input_values: dict[str, Any]) -> float:
    """Run inference using the persisted model and ordered feature columns."""
    model = load_model()
    input_frame = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_frame)[0]
    return round(float(prediction), 2)


def append_history(input_values: dict[str, Any], soh: float, condition: str) -> None:
    """Persist each prediction to a CSV file for reporting and analytics."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_PATH.exists()

    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["Timestamp", *FEATURE_COLUMNS, "PredictedSOH", "Condition"],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **input_values,
                "PredictedSOH": soh,
                "Condition": condition,
            }
        )


def read_history(limit: int | None = None) -> list[dict[str, Any]]:
    """Read prediction history with the latest entries first."""
    if not HISTORY_PATH.exists():
        return []

    history = pd.read_csv(HISTORY_PATH).tail(limit) if limit else pd.read_csv(HISTORY_PATH)
    return history.iloc[::-1].to_dict(orient="records")


def get_dataset_summary() -> dict[str, Any]:
    """Collect high-level dataset stats for dashboard cards."""
    if not DATASET_PATH.exists():
        return {"rows": 0, "avg_soh": 0, "min_soh": 0, "max_soh": 0}

    dataframe = pd.read_csv(DATASET_PATH)
    return {
        "rows": len(dataframe),
        "avg_soh": round(float(dataframe["SOH"].mean()), 2),
        "min_soh": round(float(dataframe["SOH"].min()), 2),
        "max_soh": round(float(dataframe["SOH"].max()), 2),
    }


def get_analytics_payload() -> dict[str, Any]:
    """Prepare chart-ready data from dataset and prediction history."""
    dataset = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()
    history = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()

    cycle_labels = dataset["Cycle"].head(40).tolist() if not dataset.empty else []
    soh_values = dataset["SOH"].head(40).round(2).tolist() if not dataset.empty else []

    condition_counts = (
        history["Condition"].value_counts().to_dict() if not history.empty else {}
    )

    return {
        "cycle_labels": cycle_labels,
        "soh_values": soh_values,
        "condition_labels": list(condition_counts.keys()),
        "condition_values": list(condition_counts.values()),
    }


def register_routes(app: Flask) -> None:
    """Register all page routes for the web application."""

    @app.route("/")
    def home():
        return redirect(url_for("dashboard"))

    @app.route("/dashboard")
    def dashboard():
        summary = get_dataset_summary()
        recent_predictions = read_history(limit=5)
        return render_template(
            "dashboard.html",
            active_page="dashboard",
            summary=summary,
            recent_predictions=recent_predictions,
        )

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        prediction = None
        submitted_values: dict[str, Any] = {}

        if request.method == "POST":
            submitted_values, errors = parse_prediction_form(request.form)

            if errors:
                for error in errors:
                    flash(error, "danger")
            else:
                soh = predict_soh(submitted_values)
                condition = get_battery_condition(soh)
                append_history(submitted_values, soh, condition)
                prediction = {"soh": soh, "condition": condition}
                flash("Prediction completed and saved to history.", "success")

        return render_template(
            "predict.html",
            active_page="predict",
            fields=FIELD_RULES,
            feature_columns=FEATURE_COLUMNS,
            prediction=prediction,
            submitted_values=submitted_values,
        )

    @app.route("/history")
    def history():
        return render_template(
            "history.html",
            active_page="history",
            predictions=read_history(),
        )

    @app.route("/analytics")
    def analytics():
        return render_template(
            "analytics.html",
            active_page="analytics",
            analytics=get_analytics_payload(),
        )

    @app.route("/report")
    def report():
        predictions = read_history()

        total_predictions = len(predictions)

        healthy = sum(1 for p in predictions if p["Condition"] == "Healthy")
        good = sum(1 for p in predictions if p["Condition"] == "Good")
        moderate = sum(1 for p in predictions if p["Condition"] == "Moderate")
        replace = sum(1 for p in predictions if p["Condition"] == "Replace Soon")
        summary = get_dataset_summary()
        return render_template(
    "report.html",
    active_page="report",
    summary=summary,
    predictions=predictions,
    total_predictions=total_predictions,
    healthy=healthy,
    good=good,
    moderate=moderate,
    replace=replace
)
    @app.route("/about")
    def about():
        return render_template(
            "about.html",
            active_page="about"
        )

    @app.route("/download-report")
    def download_report():
        from flask import send_file
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet

        predictions = read_history()

        pdf_path = "reports/EV_Battery_Report.pdf"

        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("<b>EV Battery Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1, 20))

        total = len(predictions)
        healthy = sum(1 for p in predictions if p["Condition"] == "Healthy")
        good = sum(1 for p in predictions if p["Condition"] == "Good")
        moderate = sum(1 for p in predictions if p["Condition"] == "Moderate")
        replace = sum(1 for p in predictions if p["Condition"] == "Replace Soon")

        elements.append(Paragraph(f"<b>Total Predictions:</b> {total}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Healthy:</b> {healthy}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Good:</b> {good}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Moderate:</b> {moderate}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Replace Soon:</b> {replace}", styles["Normal"]))
        elements.append(Spacer(1, 20))

        table_data = [
            ["Timestamp", "BatchID", "Cycle", "Voltage", "Predicted SOH", "Condition"]
        ]

        for p in predictions:
            table_data.append([
                str(p["Timestamp"]),
                str(p["BatchID"]),
                str(p["Cycle"]),
                str(p["Voltage"]),
                f'{p["PredictedSOH"]}%',
                str(p["Condition"])
            ])

        table = Table(table_data)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ]))

        elements.append(table)

        doc.build(elements)

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="EV_Battery_Report.pdf"
        )
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)