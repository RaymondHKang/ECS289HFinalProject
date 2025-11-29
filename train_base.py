import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import DEVICE, MAX_SEQ_LENGTH
from models import NSFWClassifier, get_tokenizer
#from data.dummy_data import make_dummy_classification_dataset
from data.unified_nsfw import load_unified_nsfw


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
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
):
    tokenizer = get_tokenizer()
    dataset = load_unified_nsfw(split="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b),
    )

    model = NSFWClassifier()
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
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

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    # Save for later (for concept training & RISE)
    model.model.save_pretrained("checkpoints/base_model")
    tokenizer.save_pretrained("checkpoints/base_model")


if __name__ == "__main__":
    train_base()
