import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_NAME, NUM_LABELS


class NSFWClassifier(nn.Module):
    """
    Wrapper around a HuggingFace Transformer classifier.
    Exposes logits + hidden states for downstream concept head & RISE.
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        """
        Returns:
            dict with:
              - loss (optional)
              - logits: (batch, num_labels)
              - hidden_states: tuple of layer outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        loss = outputs.loss if labels is not None else None
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }


def get_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure we have a mask token
    if tokenizer.mask_token is None:
        # For models without [MASK], we fall back to pad token as neutral
        tokenizer.mask_token = tokenizer.pad_token
    return tokenizer
