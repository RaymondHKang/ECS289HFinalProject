// A drop-in mock that simulates /infer results. Later, replace `mockInfer` with a real fetch.
import type { InferenceResponse } from './types';

const DEFAULT_CONCEPTS = ['toxicity', 'insult', 'threat', 'hate'];

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }

function seededRand(seed: number) {
  // simple LCG for reproducible demo based on text length
  let s = seed % 2147483647;
  return () => (s = s * 48271 % 2147483647) / 2147483647;
}

export async function mockInfer(text: string, concepts = DEFAULT_CONCEPTS): Promise<InferenceResponse> {
  // Simulate latency
  await new Promise(r => setTimeout(r, 200));

  const rand = seededRand(text.length + (text.charCodeAt(0) || 0));
  const toks = text.trim().length ? text.split(/\s+/) : ['Hello', 'world'];
  const sal = toks.map((t, i) => clamp01((t.length / 10) * (0.6 + rand() * 0.8)));
  const unsafe = clamp01(0.2 + 0.8 * Math.max(...sal));
  const safe = clamp01(1 - unsafe);

  const c: Record<string, number> = {};
  concepts.forEach((name, i) => { c[name] = clamp01(rand() * 0.9); });
  // slightly bias if spicy words present
  if (/kill|hate|stupid|sex|bomb|die/i.test(text)) {
    if (c['threat'] !== undefined) c['threat'] = clamp01(0.7 + 0.3 * rand());
    if (c['toxicity'] !== undefined) c['toxicity'] = clamp01(0.6 + 0.4 * rand());
  }

  return {
    safe_score: +safe.toFixed(4),
    label: safe >= 0.5 ? 'safe' : 'unsafe',
    tokens: toks,
    token_scores: sal.map(x => +x.toFixed(4)),
    concepts: c,
    token_concepts: toks.slice(0, 3).map((t, i) => ({ token: t, concept: concepts[i % concepts.length], strength: +clamp01(0.5 + rand() * 0.5).toFixed(3) }))
  };
}