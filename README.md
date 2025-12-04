# Interpretable NSFW Text Moderation via RISE + Post-hoc Concept Bottleneck Models

This repository contains an implementation of an interpretable NSFW text moderation system combining:

- A **RoBERTa-based NSFW classifier**
- A **post-hoc Concept Bottleneck Model (CBM)** for interpretable intermediate reasoning
- A **RISE text explainer** for token-level attribution
- A runnable **demo pipeline** for classification, concept extraction, and RISE explanations

---

## 📦 Project Structure

```
nsfw_rise_cbm/
  config.py
  requirements.txt
  models/
    __init__.py
    base_model.py
    concept_head.py
  explainers/
    __init__.py
    rise_text.py
  data/
    __init__.py
    dummy_data.py
  train_base.py
  train_concepts.py
  demo_inference.py
```

---

# 1. Installation

### Prerequisites
- Python 3.9+
- CUDA GPU recommended (optional but faster)

---

### Install Dependencies

```bash
git clone <your-repo>
cd nsfw_rise_cbm

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# 2. Train the Base NSFW Classifier

Fine-tune RoBERTa-base on NSFW classification:

```bash
python train_base.py
```

This will save:

```
checkpoints/base_model/
```

---

# 3. Train the Concept Bottleneck Head

After the base classifier is trained and frozen:

```bash
python train_concepts.py
```

This creates:

```
checkpoints/concept_head.pt
```

The concept head predicts interpretable concepts such as sexual content, hate speech, threats, profanity, etc.

---

# 4. Run the Demo (Classification + Concepts + RISE)

```bash
python demo_inference.py
```

Outputs:

- NSFW prediction score  
- 10-dimensional concept vector  
- RISE token importances (top influential tokens)

Example:

```
NSFW score: 0.9831
Concept scores: [0.91, 0.02, ..., 0.77]

RISE top tokens:
hate            | importance=0.1143
disgusting      | importance=0.0901
friends         | importance=0.0428
```

---

# 5. Updating Concepts

To modify concept definitions:

- Update `NUM_CONCEPTS` in `config.py`
- Edit `dummy_data.py` or load real datasets
- Re-run `train_concepts.py`

---

# 6. Using Real Datasets

Dataset loaders are in `data/`.

---

# 7. Visualization Dashboard

We visualize the project with streamlit which contains and can be run by:

```bash
streamlit run viz/dashboard.py
```

- Token heatmaps  
- Concept bar charts  
- Token × Concept bipartite visualizations  
- A Streamlit dashboard or Jupyter notebook  


---

# 8. Zipping the Project

### macOS/Linux
```bash
cd ..
zip -r nsfw_rise_cbm.zip nsfw_rise_cbm
```

### Windows PowerShell
```powershell
Compress-Archive -Path nsfw_rise_cbm -DestinationPath nsfw_rise_cbm.zip
```

---

# 9. Troubleshooting

### CUDA not available?
Set device to `"cpu"` in `config.py`.

### Model not loading?
Ensure you ran:

```
python train_base.py
python train_concepts.py
```

### Mismatch in concept dimensions?
Adjust:
- `NUM_CONCEPTS` in `config.py`
- Concept vectors in `dummy_data.py`

---

# 10. License

MIT License — free for academic or commercial use.

---
