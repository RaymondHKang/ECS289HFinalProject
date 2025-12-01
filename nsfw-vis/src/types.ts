// Shared types to keep API contract stable when you swap mock -> real backend.
export type Concepts = Record<string, number>;

export interface TokenConceptLink {
  token: string;
  concept: string;
  strength: number; // 0..1
}

export interface InferenceResponse {
  safe_score: number; // 0..1 (probability of SAFE)
  label: 'safe' | 'unsafe';
  tokens: string[];
  token_scores: number[]; // 0..1, same length as tokens
  concepts: Concepts; // e.g., { toxicity: 0.84, insult: 0.37, ... }
  token_concepts: TokenConceptLink[]; // optional advanced view
}