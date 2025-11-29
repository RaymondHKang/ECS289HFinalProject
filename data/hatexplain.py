from datasets import load_dataset
from typing import Literal

Split = Literal["train", "validation", "test"]


def load_hatexplain(split: Split = "train"):
    """
    Load HateXplain from HF and map to (text, label).

    HF dataset: "hatexplain"
    Typical fields: "text" (list of tokens), "label" (int or list).
    We'll:
        - join token list into text
        - map label to binary NSFW (non-hate/offensive vs hate/offensive)
    """
    ds = load_dataset("hatexplain")

    def map_example(example):
        # text is often a list of tokens
        text_tokens = example.get("text", [])
        if isinstance(text_tokens, list):
            text = " ".join(text_tokens)
        else:
            text = str(text_tokens)

        # HateXplain label schemas vary; often 0/1/2 or multi-label.
        # We'll treat any non-zero label as NSFW by default.
        label_raw = example.get("label", 0)
        if isinstance(label_raw, list):
            # e.g., multi-label; mark nsfw if any positive
            label = int(any(int(x) != 0 for x in label_raw))
        else:
            label = int(int(label_raw) != 0)

        return {"text": text, "label": label}

    mapped = ds.map(map_example, remove_columns=ds["train"].column_names)

    if split == "train":
        return mapped["train"]
    elif split == "validation":
        return mapped["validation"] if "validation" in mapped else mapped["train"]
    elif split == "test":
        return mapped["test"] if "test" in mapped else mapped["train"]
    else:
        raise ValueError(f"Unknown split: {split}")
