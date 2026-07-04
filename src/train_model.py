"""Train regression models for EV battery state of health prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


CLEANED_DATASET_PATH = Path("dataset/cleaned_battery_dataset.csv")
MODEL_OUTPUT_PATH = Path("models/best_model.pkl")
TARGET_COLUMN = "SOH"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(file_path):
    """Load the cleaned battery state of health dataset from a CSV file."""
    return pd.read_csv(file_path)


def split_features_and_target(dataframe, target_column):
    """Split the dataset into features and target."""
    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    return features, target


def split_train_test_data(features, target):
    """Split features and target into training and testing datasets."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def get_regression_models():
    """Create the regression models used for SOH prediction."""
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            random_state=RANDOM_STATE,
        ),
    }


def calculate_rmse(actual_values, predicted_values):
    """Calculate root mean squared error."""
    mse = mean_squared_error(actual_values, predicted_values)
    return mse ** 0.5


def evaluate_model(model, features_test, target_test):
    """Calculate evaluation metrics for a trained regression model."""
    predictions = model.predict(features_test)

    return {
        "MAE": mean_absolute_error(target_test, predictions),
        "RMSE": calculate_rmse(target_test, predictions),
        "R2 Score": r2_score(target_test, predictions),
    }


def train_and_evaluate_models(
    models,
    features_train,
    features_test,
    target_train,
    target_test,
):
    """Train each model and collect evaluation results."""
    trained_models = {}
    evaluation_results = []

    for model_name, model in models.items():
        model.fit(features_train, target_train)
        metrics = evaluate_model(model, features_test, target_test)

        trained_models[model_name] = model
        evaluation_results.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2 Score": metrics["R2 Score"],
            }
        )

    return trained_models, pd.DataFrame(evaluation_results)


def display_evaluation_results(results_dataframe):
    """Display model evaluation results in a formatted table."""
    print("\nModel Evaluation Results:")
    print(results_dataframe.to_string(index=False))


def get_best_model_name(results_dataframe):
    """Get the model name with the highest R2 score."""
    best_model_index = results_dataframe["R2 Score"].idxmax()
    return results_dataframe.loc[best_model_index, "Model"]


def display_best_model(best_model_name, results_dataframe):
    """Display the best model based on the highest R2 score."""
    best_model_score = results_dataframe.loc[
        results_dataframe["Model"] == best_model_name,
        "R2 Score",
    ].iloc[0]

    print("\nBest Model Based on Highest R2 Score:")
    print(f"{best_model_name} with R2 Score: {best_model_score}")


def save_model(model, file_path):
    """Save a trained model to a pickle file using joblib."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)


def main():
    """Run the EV battery SOH model training workflow."""
    dataframe = load_dataset(CLEANED_DATASET_PATH)
    features, target = split_features_and_target(dataframe, TARGET_COLUMN)
    features_train, features_test, target_train, target_test = (
        split_train_test_data(features, target)
    )

    # Train all candidate models and compare their regression performance.
    models = get_regression_models()
    trained_models, results_dataframe = train_and_evaluate_models(
        models,
        features_train,
        features_test,
        target_train,
        target_test,
    )

    display_evaluation_results(results_dataframe)

    # Select and persist the best model by R2 score.
    best_model_name = get_best_model_name(results_dataframe)
    display_best_model(best_model_name, results_dataframe)
    save_model(trained_models[best_model_name], MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
