"""Clean the EV battery state of health prediction dataset."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATASET_PATH = Path("dataset/battery_dataset.csv")
CLEANED_DATASET_PATH = Path("dataset/cleaned_battery_dataset.csv")
ID_COLUMN = "BatteryID"
BATCH_COLUMN = "BatchID"


def load_dataset(file_path):
    """Load the battery state of health dataset from a CSV file."""
    return pd.read_csv(file_path)


def remove_column(dataframe, column_name):
    """Remove the specified column from the dataset."""
    return dataframe.drop(columns=[column_name])


def print_encoding_mapping(column_name, label_encoder):
    """Print the original-to-encoded value mapping for a column."""
    print(f"\n{column_name} Encoding Mapping:")
    for encoded_value, original_value in enumerate(label_encoder.classes_):
        print(f"{original_value} -> {encoded_value}")


def encode_column(dataframe, column_name):
    """Encode a categorical column using LabelEncoder."""
    cleaned_dataframe = dataframe.copy()
    label_encoder = LabelEncoder()

    cleaned_dataframe[column_name] = label_encoder.fit_transform(
        cleaned_dataframe[column_name]
    )
    print_encoding_mapping(column_name, label_encoder)

    return cleaned_dataframe


def display_first_five_rows(dataframe):
    """Display the first five rows of the cleaned dataset."""
    print("\nFirst Five Rows of Cleaned Dataset:")
    print(dataframe.head())


def display_dataset_shape(dataframe):
    """Display the updated dataset shape."""
    print("\nUpdated Dataset Shape:")
    print(dataframe.shape)


def save_cleaned_dataset(dataframe, file_path):
    """Save the cleaned dataset to a CSV file."""
    dataframe.to_csv(file_path, index=False)


def main():
    """Run the battery dataset cleaning workflow."""
    dataframe = load_dataset(DATASET_PATH)

    # Remove the identifier column before model preparation.
    cleaned_dataframe = remove_column(dataframe, ID_COLUMN)

    # Encode BatchID so the categorical batch labels are numeric.
    cleaned_dataframe = encode_column(cleaned_dataframe, BATCH_COLUMN)

    display_first_five_rows(cleaned_dataframe)
    display_dataset_shape(cleaned_dataframe)
    save_cleaned_dataset(cleaned_dataframe, CLEANED_DATASET_PATH)


if __name__ == "__main__":
    main()
