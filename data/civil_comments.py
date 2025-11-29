from datasets import load_dataset, DatasetDict
from typing import Literal

Split = Literal["train", "validation", "test"]


def load_civil_comments(
    split: Split = "train",
    toxicity_threshold: float = 0.5,
) -> DatasetDict:
    """
    Load Civil Comments and map to a binary NSFW label based on toxicity.

    Returns a HuggingFace Dataset with columns:
        - "text": str
        - "label": int (0 = safe, 1 = nsfw)
    """
    # HF dataset name may be "civil_comments"; adjust if needed.
    ds = load_dataset("civil_comments")

    def map_example(example):
        # Common columns include "text" and "toxicity" (float [0,1]).
        text = example.get("text", "")
        tox = example.get("toxicity", 0.0)
        label = int(tox >= toxicity_threshold)
        return {"text": text, "label": label}

    mapped = ds.map(map_example, remove_columns=ds["train"].column_names)

    if split == "train":
        return mapped["train"]
    elif split == "validation":
        # Civil Comments often doesn't have explicit val; you can slice train
        return mapped["train"].select(range(0, min(5000, len(mapped["train"]))))
    elif split == "test":
        # Use provided test if available; fallback to a held-out part of train
        if "test" in mapped:
            return mapped["test"]
        else:
            n = len(mapped["train"])
            return mapped["train"].select(range(max(0, n - 5000), n))
    else:
        raise ValueError(f"Unknown split: {split}")
