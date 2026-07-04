"""Analyze the cleaned EV battery state of health dataset."""

from pathlib import Path

import pandas as pd


CLEANED_DATASET_PATH = Path("dataset/cleaned_battery_dataset.csv")
TARGET_COLUMN = "SOH"


def load_dataset(file_path):
    """Load the cleaned battery state of health dataset from a CSV file."""
    return pd.read_csv(file_path)


def display_dataset_overview(dataframe):
    """Display the dataset shape, column names, and data types."""
    print("\nDataset Shape:")
    print(dataframe.shape)

    print("\nColumn Names:")
    print(dataframe.columns.tolist())

    print("\nData Types:")
    print(dataframe.dtypes)


def display_soh_statistics(dataframe, target_column):
    """Display descriptive statistics for the SOH column."""
    print(f"\nDescriptive Statistics for {target_column}:")
    print(dataframe[target_column].describe())


def calculate_correlation_matrix(dataframe):
    """Calculate the correlation matrix for all numeric columns."""
    numeric_dataframe = dataframe.select_dtypes(include="number")
    return numeric_dataframe.corr()


def display_correlation_matrix(correlation_matrix):
    """Display the correlation matrix."""
    print("\nCorrelation Matrix:")
    print(correlation_matrix)


def get_target_correlations(correlation_matrix, target_column):
    """Get feature correlations with the target column."""
    return correlation_matrix[target_column].drop(labels=target_column)


def display_top_correlated_features(correlation_matrix, target_column):
    """Display the top five features most correlated with the target column."""
    target_correlations = get_target_correlations(
        correlation_matrix,
        target_column,
    )
    top_features = target_correlations.abs().sort_values(
        ascending=False
    ).head(5)

    print(f"\nTop 5 Features Most Correlated with {target_column}:")
    for feature in top_features.index:
        print(f"{feature}: {target_correlations[feature]}")


def display_highest_positive_correlation(correlation_matrix, target_column):
    """Display the feature with the highest positive target correlation."""
    target_correlations = get_target_correlations(
        correlation_matrix,
        target_column,
    )
    feature = target_correlations.idxmax()

    print(f"\nFeature with Highest Positive Correlation with {target_column}:")
    print(f"{feature}: {target_correlations[feature]}")


def display_highest_negative_correlation(correlation_matrix, target_column):
    """Display the feature with the highest negative target correlation."""
    target_correlations = get_target_correlations(
        correlation_matrix,
        target_column,
    )
    feature = target_correlations.idxmin()

    print(f"\nFeature with Highest Negative Correlation with {target_column}:")
    print(f"{feature}: {target_correlations[feature]}")


def main():
    """Run the cleaned battery dataset analysis workflow."""
    dataframe = load_dataset(CLEANED_DATASET_PATH)

    # Display basic dataset information before analyzing relationships.
    display_dataset_overview(dataframe)
    display_soh_statistics(dataframe, TARGET_COLUMN)

    # Use numeric correlations to compare features with SOH.
    correlation_matrix = calculate_correlation_matrix(dataframe)
    display_correlation_matrix(correlation_matrix)
    display_top_correlated_features(correlation_matrix, TARGET_COLUMN)
    display_highest_positive_correlation(correlation_matrix, TARGET_COLUMN)
    display_highest_negative_correlation(correlation_matrix, TARGET_COLUMN)


if __name__ == "__main__":
    main()
