import torch
import torch.nn as nn

from config import NUM_CONCEPTS


class ConceptHead(nn.Module):
    """
    Post-hoc concept bottleneck module.

    Takes frozen backbone hidden states and predicts a vector of concept scores in [0,1].
    """

    def __init__(self, hidden_dim: int, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_concepts = num_concepts

        # Simple linear head; can be made fancier (bottlenecks, sparsity penalties, etc.)
        self.proj = nn.Linear(hidden_dim, num_concepts)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: last_hidden_state, shape (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) with 1 = real token, 0 = pad

        Returns:
            concepts: (batch, num_concepts), sigmoid outputs in [0,1]
        """
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1)  # (b, seq, 1)
            masked_hidden = hidden_states * mask
            lengths = mask.sum(dim=1).clamp(min=1)  # (b,1)
            pooled = masked_hidden.sum(dim=1) / lengths

        logits = self.proj(pooled)
        concepts = torch.sigmoid(logits)
        return concepts
