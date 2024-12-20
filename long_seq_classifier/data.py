import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    """
    Clean input text by removing unwanted characters, links, and emojis.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    return text


def load_and_preprocess_data(path, label_col, text_col, save_label_encoder=True):
    """
    Load and preprocess dataset for training and validation.

    Parameters:
        path (str): Path to the CSV file.
        label_col (str): Column name for the labels.
        text_col (str): Column name for the text data.
        save_label_encoder (bool): Whether to save the label encoder as a pickle file.

    Returns:
        train_df (DataFrame): Training dataset.
        val_df (DataFrame): Validation dataset.
        num_classes (int): Number of unique classes.
    """
    df = pd.read_csv(path)

    # Clean text data
    df["x"] = df[text_col].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    df["y"] = le.fit_transform(df[label_col])
    num_classes = len(le.classes_)

    if save_label_encoder:
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, val_df, num_classes


if __name__ == "__main__":
    # Define dataset path and column names
    dataset_path = "synthetic_dataset.csv"  # Replace with your actual dataset file path
    label_column = "sentiment"  # Column name for labels in the dataset
    text_column = "review"  # Column name for text in the dataset

    # Load and preprocess the data
    train_df, val_df, num_classes = load_and_preprocess_data(
        path=dataset_path,
        label_col=label_column,
        text_col=text_column,
        save_label_encoder=True,
    )

    # Save the preprocessed datasets
    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)

    print(f"Data preprocessing complete!")
    print(f"Number of classes: {num_classes}")
    print(f"Training data saved to 'train_data.csv'.")
    print(f"Validation data saved to 'val_data.csv'.")
