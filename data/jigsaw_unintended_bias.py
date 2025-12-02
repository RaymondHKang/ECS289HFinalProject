import os
from typing import Literal

import pandas as pd
from datasets import Dataset

Split = Literal["train", "validation", "test"]


def load_jigsaw_unintended_bias(
    split: Split = "train",
    toxicity_threshold: float = 0.5,
    csv_path: str = "/Users/raymondkang/Desktop/ECS289HFall2025/newone/data/all_data.csv",
) -> Dataset:
    """
    Load Jigsaw Unintended Bias from a local CSV and map to binary NSFW.

    Expected columns in CSV:
        - 'comment_text'  (the raw text)
        - 'toxicity'      (float in [0, 1])

    Returns a HuggingFace Dataset with columns:
        - 'text':  str
        - 'label': int (0 = safe, 1 = nsfw/toxic)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Jigsaw CSV not found at {csv_path}.\n"
            "Download 'train.csv' from the Kaggle Jigsaw Unintended Bias "
            "competition and place it there, or update csv_path."
        )

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    if "comment_text" not in df.columns or "toxicity" not in df.columns:
        raise ValueError(
            "Jigsaw CSV must have 'comment_text' and 'toxicity' columns.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Keep only relevant columns
    df = df[["comment_text", "toxicity"]].copy()
    df.rename(columns={"comment_text": "text"}, inplace=True)

    # Clean toxicity and binarize
    df["toxicity"] = df["toxicity"].fillna(0.0).astype(float)
    df["label"] = (df["toxicity"] >= toxicity_threshold).astype(int)

    # Final schema: text, label
    df = df[["text", "label"]]

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)
    n = len(dataset)

    # Simple 80/10/10 split
    if split == "train":
        return dataset.select(range(0, int(0.8 * n)))
    elif split == "validation":
        return dataset.select(range(int(0.8 * n), int(0.9 * n)))
    elif split == "test":
        return dataset.select(range(int(0.9 * n), n))
    else:
        raise ValueError(f"Unknown split: {split}")
