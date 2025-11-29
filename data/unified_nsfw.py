from typing import Literal, List

from datasets import concatenate_datasets, Dataset

from .civil_comments import load_civil_comments
from .jigsaw_unintended_bias import load_jigsaw_unintended_bias
from .hatexplain import load_hatexplain

Split = Literal["train", "validation", "test"]


def load_unified_nsfw(
    split: Split = "train",
    include: List[str] = None,
) -> Dataset:
    """
    Load and unify multiple NSFW datasets into a single (text, label) dataset.

    Args:
        split: "train", "validation", or "test"
        include: list of dataset names to include, e.g.
                 ["civil_comments", "jigsaw", "hatexplain"]

    Returns:
        HuggingFace Dataset with columns:
            - "text": str
            - "label": int (0 = safe, 1 = nsfw)
    """
    if include is None:
        #include = ["civil_comments", "jigsaw", "hatexplain"]
        include = ["civil_comments", "jigsaw", "hatexplain"]

    ds_list = []

    #if "civil_comments" in include:
        #ds_list.append(load_civil_comments(split=split))

    if "jigsaw" in include:
        ds_list.append(load_jigsaw_unintended_bias(split=split))

    #if "hatexplain" in include:
        #ds_list.append(load_hatexplain(split=split))

    if not ds_list:
        raise ValueError("No datasets selected in `include`.")

    unified = concatenate_datasets(ds_list)
    unified = unified.shuffle(seed=42)
    return unified
