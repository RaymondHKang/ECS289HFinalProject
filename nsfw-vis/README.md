# Vue 3 + TypeScript + Vite

This template should help get you started developing with Vue 3 and TypeScript in Vite. The template uses Vue 3 `<script setup>` SFCs, check out the [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) to learn more.

Learn more about the recommended Project Setup and IDE Support in the [Vue Docs TypeScript Guide](https://vuejs.org/guide/typescript/overview.html#project-setup).

# NSFW Visualization Demo (Vue 3 + TypeScript + Vite)

This project is a front‑end visualization interface designed to display the outputs of an NSFW text moderation model. It currently uses mock data, but the structure is prepared for integrating a real AI backend with minimal changes.

The UI includes:
- Input text box
- Safe score indicator
- Token‑level saliency visualization
- Concept score panel
- Optional JSON inspection mode

The system is modular: all AI communication is isolated to a single file (`src/api.ts`), and the output format follows a strict interface definition (`src/types.ts`).

---

## Project Structure

```
src/
  api.ts                  # Central API entry. Replace mock with real AI calls here.
  types.ts                # Shared data schema. Backend must match these fields.
  mock/api.ts             # Mock inference generator (used before real AI is connected).
  components/
    ChatBox.vue           # Main visualization container
    TokenChips.vue        # Token saliency UI
    ConceptList.vue       # Concept score UI
  App.vue
  main.ts
index.html                # Page title and metadata
```

---

## InferenceResponse Schema

Any real AI backend must return JSON matching the following structure:

```
interface InferenceResponse {
  safe_score: number;              // 0..1 probability of SAFE
  label: "safe" | "unsafe";        // classification label
  tokens: string[];                // tokenized input
  token_scores: number[];          // saliency per token (0..1)
  concepts: Record<string, number>; // concept activations
  token_concepts: { token: string; concept: string; strength: number; }[];
}
```

The front‑end visualizations rely on these fields. Any backend implementation must supply values that conform to this format.

---

## How To Connect a Real AI Backend

All modifications occur in a single file: `src/api.ts`.

### 1. Current mock implementation

```
import { mockInfer } from "./mock/api";
export async function infer(text: string) {
  return mockInfer(text);
}
```

### 2. Replace with an actual HTTP request

```
export async function infer(text: string) {
  const res = await fetch(import.meta.env.VITE_API_BASE + "/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!res.ok) {
    throw new Error("Backend error: " + res.status);
  }
  return await res.json();
}
```

### 3. Configure backend address

Create a `.env` file in the project root:

```
VITE_API_BASE=http://localhost:8000
```

Restart the dev server after modifying environment variables.

---

## Backend Requirements

A backend implementation (e.g., FastAPI) must:
1. Accept POST requests at the `/infer` endpoint.
2. Read a JSON payload of the form `{ "text": "..." }`.
3. Produce a response matching the `InferenceResponse` structure.

Example backend pseudocode:

```
POST /infer
Input:  { "text": "sample input" }
Output: {
  "safe_score": 0.83,
  "label": "safe",
  "tokens": [...],
  "token_scores": [...],
  "concepts": {...},
  "token_concepts": [...]
}
```

Each field should be derived from your actual model:
- `safe_score`: model’s probability output
- `label`: thresholding of safe_score
- `tokens`: tokenizer output
- `token_scores`: saliency or explanation values (e.g., RISE, IG, attention)
- `concepts`: concept‑level classifier outputs
- `token_concepts`: optional mapping between tokens and concepts

---

## Adding or Modifying AI‑Driven Features

### Input Text
Frontend sends the raw text to the backend via `infer(text)`.  
If the backend needs additional parameters (e.g., explanation method), expose them as new fields in the POST body.

### Safe Score
Displayed in the UI from `resp.safe_score`.  
Ensure your backend provides a normalized value between 0 and 1.

### Token Saliency
Rendered from `resp.token_scores`.  
If your model uses a different explanation method, adapt the backend to convert its output into an array aligned with `resp.tokens`.

### Concepts
The concept panel reads from `resp.concepts`.  
You may expand, rename, or restructure concepts as long as the backend returns a dictionary of numeric values.

### Token–Concept Links
Optional.  
If implemented, return an array of objects describing which tokens contribute to which concepts.  
Frontend already accepts this structure.

---

## Deployment Notes

- All communication with AI happens through `src/api.ts`.  
- Ensure backend supports CORS when serving locally or from a remote host.
- When switching between mock and real AI, only update the implementation of `infer()`.

---

## Summary

This project provides a visualization interface that can operate independently of an AI model.  
To connect a real backend model, only the `infer()` function in `src/api.ts` must be updated, and the backend must return results matching the `InferenceResponse` schema.