"""Check the Battery_Status distribution in the original dataset."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path("dataset/ev_battery_degradation_v1.csv")
TARGET_COLUMN = "Battery_Status"
REPLACE_REQUIRED_STATUS = "Replace Required"


def load_dataset(file_path):
    """Load the original EV battery degradation dataset from a CSV file."""
    return pd.read_csv(file_path)


def display_total_rows(dataframe):
    """Display the total number of rows in the dataset."""
    print("\nTotal Number of Rows:")
    print(len(dataframe))


def display_status_counts(dataframe, target_column):
    """Display value counts for the target column."""
    print(f"\nValue Counts for {target_column}:")
    print(dataframe[target_column].value_counts())


def display_status_percentages(dataframe, target_column):
    """Display the percentage of each target class."""
    percentages = dataframe[target_column].value_counts(normalize=True) * 100

    print(f"\nPercentage of Each {target_column} Class:")
    print(percentages)


def display_replace_required_rows(dataframe, target_column, target_value):
    """Display all rows where the target column matches the target value."""
    replace_required_rows = dataframe[dataframe[target_column] == target_value]

    print(f'\nRows Where {target_column} is "{target_value}":')
    print(replace_required_rows)


def main():
    """Run the dataset status check workflow."""
    dataframe = load_dataset(DATASET_PATH)

    display_total_rows(dataframe)
    display_status_counts(dataframe, TARGET_COLUMN)
    display_status_percentages(dataframe, TARGET_COLUMN)
    display_replace_required_rows(
        dataframe,
        TARGET_COLUMN,
        REPLACE_REQUIRED_STATUS,
    )


if __name__ == "__main__":
    main()