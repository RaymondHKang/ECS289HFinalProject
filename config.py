# config.py

import torch

# Smaller, faster model than roberta-base
MODEL_NAME = "distilroberta-base"

# Binary NSFW classification
NUM_LABELS = 2

# Number of interpretable concepts for the CBM
NUM_CONCEPTS = 10

# Shorter sequences = faster training
MAX_SEQ_LENGTH = 128

# Use Apple M1 GPU via MPS if available
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Training-size + speed knobs
TRAIN_SAMPLES = 20_000          # max training examples for base classifier
VAL_SAMPLES = 2_000             # max validation examples
CONCEPT_TRAIN_SAMPLES = 5_000   # max examples to train concept head

EPOCHS_BASE = 2                 # base NSFW model epochs
EPOCHS_CONCEPT = 5              # concept head epochs

BATCH_SIZE_BASE = 8             # batch size for base model
BATCH_SIZE_CONCEPT = 16         # batch size for concept head
