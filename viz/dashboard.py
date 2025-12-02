# viz/dashboard.py
import os, sys

# Ensure project root is on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, MAX_SEQ_LENGTH, NUM_CONCEPTS
from models.concept_head import ConceptHead
from explainers import RISETextExplainer
from concept_config import CONCEPT_NAMES


# ---------- Model loading ----------

@st.cache_resource(show_spinner=True)
def load_backbone_and_tokenizer():
    backbone = AutoModelForSequenceClassification.from_pretrained(
        "checkpoints/base_model",
        output_hidden_states=True,
    ).to(DEVICE)
    backbone.eval()

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/base_model", use_fast=True)
    return backbone, tokenizer


@st.cache_resource(show_spinner=True)
def load_concept_head(hidden_dim: int):
    head = ConceptHead(hidden_dim=hidden_dim, num_concepts=NUM_CONCEPTS)
    state = torch.load("checkpoints/concept_head.pt", map_location=DEVICE)
    head.load_state_dict(state)
    head.to(DEVICE)
    head.eval()
    return head


# ---------- Inference helpers ----------

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
    probs = torch.softmax(logits, dim=-1)[0]  # (num_labels,)
    nsfw_prob = probs[1].item()  # assuming label 1 = NSFW

    last_hidden = outputs.hidden_states[-1]  # (1, seq, hidden)
    concept_scores = concept_head(last_hidden, attention_mask=attention_mask)[0]
    concept_scores = concept_scores.detach().cpu().numpy()

    return nsfw_prob, concept_scores


def run_rise(backbone, tokenizer, text: str, num_masks: int, p_keep: float):
    from models.base_model import NSFWClassifier

    wrapper = NSFWClassifier()
    wrapper.model = backbone  # reuse weights

    explainer = RISETextExplainer(
        model=wrapper,
        tokenizer=tokenizer,
        target_label_idx=1,
        num_masks=num_masks,
        p_keep=p_keep,
        max_length=MAX_SEQ_LENGTH,
    )
    explanation = explainer.explain(text)
    return explanation["tokens"], explanation["importances"], explanation["score"]


# ---------- Visualization helpers ----------

def render_decision_card(nsfw_prob: float):
    label = "NSFW" if nsfw_prob >= 0.5 else "SAFE"
    st.subheader("Decision")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Label", label)
    col2.metric("NSFW Probability", f"{nsfw_prob:.3f}")


def render_token_heatmap(tokens, importances):
    st.subheader("Token Saliency (RISE)")

    # Normalize importances to [0,1]
    imp = np.maximum(importances, 0)
    if imp.max() > 0:
        imp = imp / imp.max()
    else:
        imp = np.zeros_like(imp)

    # Build HTML with background intensity
    html_tokens = []
    for tok, w in zip(tokens, imp):
        # simple red-ish background intensity
        color = f"rgba(255, 0, 0, {w:.2f})"
        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        span = f'<span style="background-color:{color}; padding:2px; margin:1px; border-radius:3px;">{safe_tok}</span>'
        html_tokens.append(span)

    html = "<div style='line-height:2.0;'>" + " ".join(html_tokens) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_concept_bar_chart(concept_scores):
    st.subheader("Concept Activations")

    k = min(len(CONCEPT_NAMES), len(concept_scores))
    names = CONCEPT_NAMES[:k]
    scores = concept_scores[:k]

    df = pd.DataFrame({"concept": names, "score": scores})
    df = df.set_index("concept")

    st.bar_chart(df)


def render_token_concept_heatmap(tokens, importances, concept_scores):
    st.subheader("Token × Concept Matrix (Toy Approximation)")

    # Normalize both to [0,1]
    token_imp = np.maximum(importances, 0)
    if token_imp.max() > 0:
        token_imp = token_imp / token_imp.max()
    concept_imp = np.maximum(concept_scores, 0)
    if concept_imp.max() > 0:
        concept_imp = concept_imp / concept_imp.max()

    k = min(len(CONCEPT_NAMES), len(concept_imp))
    concept_imp = concept_imp[:k]
    concepts = CONCEPT_NAMES[:k]

    # Toy matrix: outer product of token saliency and concept activation
    mat = np.outer(concept_imp, token_imp)  # shape (k, seq_len)

    # Downsample tokens if there are too many
    max_tokens_show = 40
    if len(tokens) > max_tokens_show:
        indices = np.linspace(0, len(tokens) - 1, max_tokens_show).astype(int)
        mat = mat[:, indices]
        tokens_show = [tokens[i] for i in indices]
    else:
        tokens_show = tokens

    fig, ax = plt.subplots(figsize=(min(12, 0.3 * len(tokens_show)), 0.4 * k + 2))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(range(len(tokens_show)))
    ax.set_xticklabels(tokens_show, rotation=90, fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels(concepts, fontsize=9)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Concepts")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

    st.pyplot(fig)


# ---------- Streamlit app ----------

def main():
    st.title("Interpretable NSFW Text Moderation (RISE + CBM)")

    st.sidebar.header("RISE Settings")
    num_masks = st.sidebar.slider("Number of RISE masks", min_value=32, max_value=512, value=128, step=32)
    p_keep = st.sidebar.slider("Token keep probability", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.sidebar.header("Info")
    st.sidebar.write(f"Device: `{DEVICE}`")
    st.sidebar.write(f"Max seq length: `{MAX_SEQ_LENGTH}`")

    default_text = "I hate you and your disgusting friends."
    text = st.text_area("Input text", value=default_text, height=120)

    if st.button("Run Moderation"):
        if not text.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Loading models..."):
            backbone, tokenizer = load_backbone_and_tokenizer()
            hidden_dim = backbone.config.hidden_size
            concept_head = load_concept_head(hidden_dim)

        with st.spinner("Running classifier + concept head..."):
            nsfw_prob, concept_scores = predict_with_concepts(backbone, concept_head, tokenizer, text)

        with st.spinner("Running RISE explanation..."):
            tokens, importances, rise_score = run_rise(backbone, tokenizer, text, num_masks, p_keep)

        st.markdown("---")
        render_decision_card(nsfw_prob)

        col1, col2 = st.columns(2)
        with col1:
            render_token_heatmap(tokens, importances)
        with col2:
            render_concept_bar_chart(concept_scores)

        st.markdown("---")
        render_token_concept_heatmap(tokens, importances, concept_scores)

if __name__ == "__main__":
    main()
