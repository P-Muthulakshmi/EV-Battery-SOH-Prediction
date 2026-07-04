"""Predict EV battery state of health using the trained model."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/best_model.pkl")
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


@dataclass(frozen=True)
class InputRule:
    """Validation rule for a single model input feature."""

    data_type: type
    description: str
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[int, ...] | None = None
    greater_than: float | None = None


INPUT_RULES = {
    "BatchID": InputRule(
        data_type=int,
        description="allowed values: 0, 1, 2, 3",
        allowed_values=(0, 1, 2, 3),
    ),
    "Cycle": InputRule(
        data_type=int,
        description="range: 1 to 2000",
        minimum=1,
        maximum=2000,
    ),
    "Voltage": InputRule(
        data_type=float,
        description="range: 3.0 to 4.2",
        minimum=3.0,
        maximum=4.2,
    ),
    "Current": InputRule(
        data_type=float,
        description="range: 0.5 to 2.0",
        minimum=0.5,
        maximum=2.0,
    ),
    "Temperature": InputRule(
        data_type=float,
        description="range: 10 to 41",
        minimum=10,
        maximum=41,
    ),
    "ChargeTime": InputRule(
        data_type=float,
        description="must be greater than 0",
        greater_than=0,
    ),
    "DischargeTime": InputRule(
        data_type=float,
        description="must be greater than 0",
        greater_than=0,
    ),
    "InternalResistance": InputRule(
        data_type=float,
        description="range: 0.04 to 0.06",
        minimum=0.04,
        maximum=0.06,
    ),
    "Capacity": InputRule(
        data_type=float,
        description="range: 1.46 to 2.52",
        minimum=1.46,
        maximum=2.52,
    ),
    "AmbientHumidity": InputRule(
        data_type=float,
        description="range: 30 to 70",
        minimum=30,
        maximum=70,
    ),
    "C_Rate": InputRule(
        data_type=float,
        description="range: 0.5 to 2.0",
        minimum=0.5,
        maximum=2.0,
    ),
}


def load_model(file_path):
    """Load the trained model from a pickle file."""
    return joblib.load(file_path)


def convert_value(raw_value, data_type):
    """Convert the raw user input to the required data type."""
    if data_type is int:
        if not raw_value.strip().lstrip("-").isdigit():
            raise ValueError
        return int(raw_value)

    return float(raw_value)


def is_valid_value(value, rule):
    """Return True when a value satisfies its validation rule."""
    if rule.allowed_values is not None and value not in rule.allowed_values:
        return False

    if rule.minimum is not None and value < rule.minimum:
        return False

    if rule.maximum is not None and value > rule.maximum:
        return False

    if rule.greater_than is not None and value <= rule.greater_than:
        return False

    return True


def get_validated_input(column_name, rule):
    """Ask the user for one validated feature value."""
    value_type = "integer" if rule.data_type is int else "float"

    while True:
        print(f"\n{column_name} valid {rule.description}")
        raw_value = input(f"Enter {column_name} ({value_type}): ")

        try:
            value = convert_value(raw_value, rule.data_type)
        except ValueError:
            print(
                f"Invalid {column_name}. Please enter a valid {value_type}; "
                f"{rule.description}."
            )
            continue

        if is_valid_value(value, rule):
            return value

        print(f"Invalid {column_name}. Allowed {rule.description}.")


def get_user_input():
    """Collect battery feature values from the user."""
    input_values = {}

    print("Enter battery details for SOH prediction:")
    for column in FEATURE_COLUMNS:
        input_values[column] = get_validated_input(column, INPUT_RULES[column])

    return input_values


def create_input_dataframe(input_values):
    """Create a pandas DataFrame from user-entered values."""
    return pd.DataFrame([input_values], columns=FEATURE_COLUMNS)


def predict_soh(model, input_dataframe):
    """Predict the battery SOH using the trained model."""
    prediction = model.predict(input_dataframe)
    return prediction[0]


def get_battery_condition(predicted_soh):
    """Return the battery condition based on the predicted SOH."""
    if predicted_soh >= 90:
        return "Healthy"
    if predicted_soh >= 80:
        return "Good"
    if predicted_soh >= 70:
        return "Moderate"
    return "Replace Soon"


def display_prediction(predicted_soh, battery_condition):
    """Display the predicted SOH and battery condition."""
    print(f"\nPredicted SOH: {predicted_soh:.2f}")
    print(f"Battery Condition: {battery_condition}")


def main():
    """Run the battery SOH prediction workflow."""
    model = load_model(MODEL_PATH)

    # Collect user input in the same feature order used during training.
    input_values = get_user_input()
    input_dataframe = create_input_dataframe(input_values)

    predicted_soh = predict_soh(model, input_dataframe)
    battery_condition = get_battery_condition(predicted_soh)
    display_prediction(predicted_soh, battery_condition)


if __name__ == "__main__":
    main()
