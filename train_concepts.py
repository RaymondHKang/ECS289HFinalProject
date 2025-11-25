import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, MAX_SEQ_LENGTH, NUM_CONCEPTS
from models.concept_head import ConceptHead
from data.dummy_data import make_dummy_concept_dataset


def collate_fn(tokenizer, batch):
    texts = [x["text"] for x in batch]
    concepts = torch.tensor([x["concepts"] for x in batch], dtype=torch.float)

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
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
):
    # Load frozen backbone
    backbone = AutoModelForSequenceClassification.from_pretrained(
        "checkpoints/base_model",
        output_hidden_states=True,
    ).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/base_model", use_fast=True)

    dataset = make_dummy_concept_dataset()
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
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            concepts = batch["concepts"].to(DEVICE)

            with torch.no_grad():
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden = outputs.hidden_states[-1]  # (b, seq, hidden)

            pred_concepts = concept_head(
                hidden_states=last_hidden,
                attention_mask=attention_mask,
            )

            # If NUM_CONCEPTS > provided concept dims, slice or pad.
            if pred_concepts.size(1) != concepts.size(1):
                # For demo, align by min dimension.
                k = min(pred_concepts.size(1), concepts.size(1))
                loss = bce(pred_concepts[:, :k], concepts[:, :k])
            else:
                loss = bce(pred_concepts, concepts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | Concept loss={avg_loss:.4f}")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(concept_head.state_dict(), "checkpoints/concept_head.pt")
    print("Saved concept head to checkpoints/concept_head.pt")


if __name__ == "__main__":
    train_concept_head()
