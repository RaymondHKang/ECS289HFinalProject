# train_concepts.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    DEVICE,
    MAX_SEQ_LENGTH,
    NUM_CONCEPTS,
    CONCEPT_TRAIN_SAMPLES,
    EPOCHS_CONCEPT,
    BATCH_SIZE_CONCEPT,
)
from models.concept_head import ConceptHead
from data.unified_nsfw import load_unified_nsfw  # or your concept dataset


LOG_EVERY = 50  # print concept-loss every N steps


def collate_fn(tokenizer, batch):
    texts = [x["text"] for x in batch]

    # For now, use the NSFW label as a simple one-dimensional concept.
    # Replace this with real multi-label concept vectors when you have them.
    concepts = torch.tensor([[ex["label"]] for ex in batch], dtype=torch.float)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    enc["concepts"] = concepts
    return enc


def train_concept_head(
    epochs: int = EPOCHS_CONCEPT,
    batch_size: int = BATCH_SIZE_CONCEPT,
    lr: float = 1e-3,
):
    os.makedirs("checkpoints", exist_ok=True)

    # Load frozen backbone
    backbone = AutoModelForSequenceClassification.from_pretrained(
        "checkpoints/base_model",
        output_hidden_states=True,
    ).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/base_model", use_fast=True)

    full = load_unified_nsfw(split="train")
    dataset = full.shuffle(seed=42).select(
        range(min(CONCEPT_TRAIN_SAMPLES, len(full)))
    )

    print(f"[INFO] Concept training samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )

    hidden_dim = backbone.config.hidden_size
    concept_head = ConceptHead(hidden_dim=hidden_dim, num_concepts=NUM_CONCEPTS).to(DEVICE)

    optimizer = torch.optim.AdamW(concept_head.parameters(), lr=lr)
    bce = torch.nn.BCELoss()

    for epoch in range(epochs):
        concept_head.train()
        total_loss = 0.0

        print(f"\n===== Concept Epoch {epoch + 1}/{epochs} =====")
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            concepts = batch["concepts"].to(DEVICE)

            with torch.no_grad():
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden = outputs.hidden_states[-1]

            pred_concepts = concept_head(
                hidden_states=last_hidden,
                attention_mask=attention_mask,
            )

            # Align first dimension if NUM_CONCEPTS > target dims
            k = min(pred_concepts.size(1), concepts.size(1))
            loss = bce(pred_concepts[:, :k], concepts[:, :k])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % LOG_EVERY == 0 or (step + 1) == len(loader):
                avg_step_loss = total_loss / (step + 1)
                print(
                    f"[Concept Epoch {epoch + 1}/{epochs}] "
                    f"Step {step + 1}/{len(loader)} "
                    f"Loss {loss.item():.4f} "
                    f"AvgLoss {avg_step_loss:.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"[Concept Epoch {epoch + 1}] Training loss: {avg_loss:.4f}")

        # Per-epoch checkpoint
        ckpt_path = f"checkpoints/concept_head_epoch_{epoch + 1}.pt"
        torch.save(concept_head.state_dict(), ckpt_path)
        print(f"[CHECKPOINT] Saved concept head checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = "checkpoints/concept_head.pt"
    torch.save(concept_head.state_dict(), final_path)
    print(f"\n[CHECKPOINT] Saved final concept head to: {final_path}")


if __name__ == "__main__":
    train_concept_head()
