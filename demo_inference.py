import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, MAX_SEQ_LENGTH, NUM_CONCEPTS
from models.concept_head import ConceptHead
from explainers import RISETextExplainer
from models.base_model import NSFWClassifier


def load_models():
    # Backbone classifier
    backbone = AutoModelForSequenceClassification.from_pretrained(
        "checkpoints/base_model",
        output_hidden_states=True,
    ).to(DEVICE)
    backbone.eval()

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/base_model", use_fast=True)

    # Concept head
    hidden_dim = backbone.config.hidden_size
    concept_head = ConceptHead(hidden_dim=hidden_dim, num_concepts=NUM_CONCEPTS)
    concept_head.load_state_dict(torch.load("checkpoints/concept_head.pt", map_location=DEVICE))
    concept_head.to(DEVICE)
    concept_head.eval()

    return backbone, concept_head, tokenizer


@torch.no_grad()
def predict_with_concepts(backbone, concept_head, tokenizer, text: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    nsfw_score = probs[0, 1].item()  # assuming label 1 = NSFW

    last_hidden = outputs.hidden_states[-1]
    concepts = concept_head(last_hidden, attention_mask=attention_mask)[0]  # (num_concepts,)
    concepts = concepts.cpu().numpy()

    return nsfw_score, concepts


def main():
    backbone, concept_head, tokenizer = load_models()

    text = "I hate you and your disgusting friends."  # example input

    print("Input:", text)

    nsfw_score, concepts = predict_with_concepts(backbone, concept_head, tokenizer, text)
    print(f"NSFW score: {nsfw_score:.4f}")
    print("Concept scores:", concepts)

    # Wrap backbone in NSFWClassifier interface for RISE
    wrapper = NSFWClassifier()
    wrapper.model = backbone

    explainer = RISETextExplainer(
        model=wrapper,
        tokenizer=tokenizer,
        target_label_idx=1,
        num_masks=300,      # fewer for quick demo
        p_keep=0.5,
        max_length=MAX_SEQ_LENGTH,
    )

    explanation = explainer.explain(text)
    tokens = explanation["tokens"]
    importances = explanation["importances"]
    score = explanation["score"]

    print(f"\nRISE NSFW score (original): {score:.4f}")
    print("Top tokens by importance:")
    top_idx = np.argsort(-importances)[:10]
    for idx in top_idx:
        print(f"{tokens[idx]:15s} | importance={importances[idx]:.4f}")


if __name__ == "__main__":
    main()
