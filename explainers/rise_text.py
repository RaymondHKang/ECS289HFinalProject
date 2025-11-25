import torch
import numpy as np
from typing import Dict

from config import DEVICE, MAX_SEQ_LENGTH


class RISETextExplainer:
    """
    RISE for text: random token masking + score aggregation.
    Works with any black-box classifier that returns logits.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_label_idx: int = 1,
        num_masks: int = 500,
        p_keep: float = 0.5,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        """
        Args:
            model: NSFWClassifier (or any HF model wrapper with .forward returning logits)
            tokenizer: corresponding tokenizer
            target_label_idx: index of label to explain (e.g., NSFW = 1)
            num_masks: number of random masks to sample
            p_keep: probability of keeping a token (vs masking)
            max_length: max sequence length for tokenization
        """
        self.model = model.to(DEVICE)
        self.model.eval()
        self.tokenizer = tokenizer
        self.target_label_idx = target_label_idx
        self.num_masks = num_masks
        self.p_keep = p_keep
        self.max_length = max_length

    @torch.no_grad()
    def explain(self, text: str) -> Dict:
        """
        Returns:
            {
              "tokens": List[str],
              "importances": np.array of shape (seq_len,),
              "score": float (original NSFW score),
            }
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoding["input_ids"].to(DEVICE)          # (1, seq_len)
        attention_mask = encoding["attention_mask"].to(DEVICE)  # (1, seq_len)

        seq_len = input_ids.size(1)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Compute original score
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # (1, num_labels)
        probs = torch.softmax(logits, dim=-1)
        base_score = probs[0, self.target_label_idx].item()

        # Special tokens should not be masked
        special_ids = set(self.tokenizer.all_special_ids)
        token_is_special = torch.tensor(
            [tid.item() in special_ids for tid in input_ids[0]],
            device=DEVICE,
        )  # (seq_len,)

        # Initialize importance accumulator
        importances = torch.zeros(seq_len, device=DEVICE)

        # Sample random masks
        for _ in range(self.num_masks):
            # 1 = keep, 0 = mask
            mask = torch.bernoulli(
                torch.full((seq_len,), self.p_keep, device=DEVICE)
            ).long()

            # Always keep special tokens
            mask[token_is_special] = 1

            masked_input_ids = input_ids.clone()

            # Replace masked tokens with [MASK] or pad
            mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.pad_token_id
            masked_input_ids[0, mask == 0] = mask_token_id

            outputs_m = self.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
            )
            logits_m = outputs_m["logits"]
            probs_m = torch.softmax(logits_m, dim=-1)
            score_m = probs_m[0, self.target_label_idx]

            # Add contribution to kept tokens
            importances += score_m * mask

        # Normalize importances
        importances = importances / self.num_masks
        importances = importances.cpu().numpy()

        return {
            "tokens": tokens,
            "importances": importances,
            "score": base_score,
        }
