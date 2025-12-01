// Central API shim. Later, replace the body of `infer` to call your FastAPI endpoint.
import type { InferenceResponse } from './types';
import { mockInfer } from './mock/api';

export async function infer(text: string): Promise<InferenceResponse> {
  // TODO (when ready):
  // const r = await fetch('http://localhost:8000/infer', {
  //   method: 'POST', headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ text })
  // });
  // return await r.json();
  return mockInfer(text);
}