"""Initial data inspection for EV battery state of health prediction."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path("dataset/battery_dataset.csv")
SOH_COLUMN = "SOH"
BATCH_COLUMN = "BatchID"


def load_dataset(file_path):
    """Load the battery state of health dataset from a CSV file."""
    return pd.read_csv(file_path)


def display_first_five_rows(dataframe):
    """Display the first five rows of the dataset."""
    print("\nFirst Five Rows:")
    print(dataframe.head())


def display_dataset_shape(dataframe):
    """Display the number of rows and columns in the dataset."""
    print("\nDataset Shape:")
    print(dataframe.shape)


def display_column_names(dataframe):
    """Display all column names in the dataset."""
    print("\nColumn Names:")
    print(dataframe.columns.tolist())


def display_data_types(dataframe):
    """Display the data type of each column."""
    print("\nData Types:")
    print(dataframe.dtypes)


def display_missing_values(dataframe):
    """Display the number of missing values in each column."""
    print("\nMissing Values:")
    print(dataframe.isnull().sum())


def display_duplicate_rows(dataframe):
    """Display the number of duplicate rows in the dataset."""
    print("\nDuplicate Rows:")
    print(dataframe.duplicated().sum())


def display_summary_statistics(dataframe):
    """Display summary statistics for numeric columns."""
    print("\nSummary Statistics:")
    print(dataframe.describe())


def display_unique_values(dataframe, column_name):
    """Display unique values for the specified column."""
    print(f"\nUnique Values for {column_name}:")
    print(dataframe[column_name].unique())


def display_soh_statistics(dataframe, column_name):
    """Display minimum, maximum, and mean SOH values."""
    print("\nSOH Statistics:")
    print(f"Minimum SOH: {dataframe[column_name].min()}")
    print(f"Maximum SOH: {dataframe[column_name].max()}")
    print(f"Mean SOH: {dataframe[column_name].mean()}")


def main():
    """Run the data preprocessing inspection workflow."""
    dataframe = load_dataset(DATASET_PATH)

    # Display core dataset information for initial inspection.
    display_first_five_rows(dataframe)
    display_dataset_shape(dataframe)
    display_column_names(dataframe)
    display_data_types(dataframe)
    display_missing_values(dataframe)
    display_duplicate_rows(dataframe)
    display_summary_statistics(dataframe)

    # Display project-specific categorical and target information.
    display_unique_values(dataframe, BATCH_COLUMN)
    display_soh_statistics(dataframe, SOH_COLUMN)


if __name__ == "__main__":
    main()
