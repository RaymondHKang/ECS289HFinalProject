MODEL_NAME = "roberta-base"

NUM_LABELS = 2           # NSFW vs SFW (binary)
NUM_CONCEPTS = 10        # e.g., sexual, insult, threat, profanity, self-harm, etc.
MAX_SEQ_LENGTH = 256

DEVICE = "cuda"  # or "cpu"
