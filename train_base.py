# train_base.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import (
    DEVICE,
    MAX_SEQ_LENGTH,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    EPOCHS_BASE,
    BATCH_SIZE_BASE,
)
from models import NSFWClassifier, get_tokenizer
from data.unified_nsfw import load_unified_nsfw


LOG_EVERY = 50  # print training loss every N steps


def collate_fn(tokenizer, batch):
    texts = [x["text"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def train_base(
    epochs: int = EPOCHS_BASE,
    batch_size: int = BATCH_SIZE_BASE,
    lr: float = 2e-5,
):
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer = get_tokenizer()

    # Load unified NSFW dataset
    full_train = load_unified_nsfw(split="train")
    full_val = load_unified_nsfw(split="validation")

    # Subsample for speed
    train = full_train.shuffle(seed=42).select(
        range(min(TRAIN_SAMPLES, len(full_train)))
    )
    val = full_val.shuffle(seed=42).select(
        range(min(VAL_SAMPLES, len(full_val)))
    )

    print(f"[INFO] Train samples: {len(train)}, Val samples: {len(val)}")

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )

    model = NSFWClassifier()
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if (step + 1) % LOG_EVERY == 0 or (step + 1) == len(train_loader):
                avg_step_loss = total_loss / (step + 1)
                print(
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"Step {step + 1}/{len(train_loader)} "
                    f"Global {global_step} "
                    f"Loss {loss.item():.4f} "
                    f"AvgLoss {avg_step_loss:.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Training loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        if total > 0:
            acc = correct / total
        else:
            acc = 0.0
        print(f"[Epoch {epoch + 1}] Val acc: {acc:.4f}")

        # Save per-epoch checkpoint
        epoch_ckpt_dir = f"checkpoints/base_model_epoch_{epoch + 1}"
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        model.model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)
        print(f"[CHECKPOINT] Saved base model checkpoint: {epoch_ckpt_dir}")

    # Final / best checkpoint (you can change logic later)
    final_dir = "checkpoints/base_model"
    os.makedirs(final_dir, exist_ok=True)
    model.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[CHECKPOINT] Saved final base model to: {final_dir}")


if __name__ == "__main__":
    train_base()
