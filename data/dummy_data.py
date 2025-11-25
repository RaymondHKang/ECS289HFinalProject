from datasets import Dataset


def make_dummy_classification_dataset() -> Dataset:
    """
    Toy dataset for quick sanity checks.
    Replace this with real dataset loading & preprocessing.
    """
    texts = [
        "You are a wonderful person.",
        "I hate you and your disgusting friends.",
        "This is a benign comment about cooking pasta.",
        "That group is trash and should disappear.",
    ]
    # 0 = safe, 1 = nsfw/toxic
    labels = [0, 1, 0, 1]

    return Dataset.from_dict({"text": texts, "label": labels})


def make_dummy_concept_dataset() -> Dataset:
    """
    Each example has concept labels (multi-label).
    Here we define 3 example concepts: insult, hate, profanity.
    You can generalize to NUM_CONCEPTS and align indices.
    """
    texts = [
        "You are a wonderful person.",
        "I hate you and your disgusting friends.",
        "This is a benign comment about cooking pasta.",
        "That group is trash and should disappear.",
    ]
    # insult, hate, profanity (binary vectors)
    concepts = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
    ]
    return Dataset.from_dict({"text": texts, "concepts": concepts})
