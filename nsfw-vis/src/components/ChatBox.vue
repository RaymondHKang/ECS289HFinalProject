<template>
  <div class="wrapper">
    <header class="title">NSFW Checker</header>

    <div class="card">
      <div class="left">
        <label class="label">Input Text:</label>
        <textarea v-model="input" class="textbox" placeholder="Type here…" />

        <div class="row">
          <button class="btn" @click="onSample">Check</button>
          <div v-if="busy" class="spinner"/>
          <div v-if="resp" class="badge">Safe Score: {{ fmt(resp.safe_score) }}</div>
        </div>

        <TokenChips v-if="resp" :tokens="resp.tokens" :scores="resp.token_scores" />

        <div class="footer">
          <button class="link" @click="toggleAdvanced">{{ showAdvanced ? 'Hide' : 'Show' }} Advanced</button>
          <button class="link" @click="copyJson">Copy JSON</button>
        </div>
        <pre v-if="showAdvanced && resp" class="json">{{ pretty(resp) }}</pre>
      </div>

      <div class="right" v-if="resp">
        <ConceptList :concepts="resp.concepts" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { infer } from '../api';
import type { InferenceResponse } from '../types';
import TokenChips from './TokenChips.vue';
import ConceptList from './ConceptList.vue';

const input = ref('I will kill you');
const resp = ref<InferenceResponse | null>(null);
const busy = ref(false);
const showAdvanced = ref(false);

const fmt = (x: number) => x.toFixed(2);
const pretty = (o: unknown) => JSON.stringify(o, null, 2);

async function onSample() {
  busy.value = true;
  try {
    resp.value = await infer(input.value);
  } finally {
    busy.value = false;
  }
}

function toggleAdvanced() { showAdvanced.value = !showAdvanced.value; }
async function copyJson() {
  if (!resp.value) return;
  await navigator.clipboard.writeText(JSON.stringify(resp.value));
}
</script>

<style scoped>
.wrapper { max-width: 980px; margin: 40px auto; font-family: Inter, ui-sans-serif; }
.title { font-size: 28px; font-weight: 800; color: #00c6cf; background:#08294a; padding: 14px 18px; border-radius: 8px; }
.card { display: grid; grid-template-columns: 1fr 300px; gap: 22px; margin-top: 14px; padding: 18px; border: 2px solid #e6e6e6; border-radius: 14px; box-shadow: 0 4px 20px rgba(0,0,0,.05); }
.left { display:flex; flex-direction: column; gap: 12px; }
.right { display:flex; }
.textbox { min-height: 110px; border: 4px solid #8c8c8c; border-radius: 12px; font-size: 16px; padding: 10px; }
.label { font-weight: 700; color: #333; display:block; margin-bottom: 4px; }
.row { display:flex; align-items:center; gap: 10px; }
.btn { background:#ffffff; border:2px solid #d3d3d3; border-radius: 10px; padding: 6px 12px; cursor:pointer; }
.badge { background:#ff8a00; color:#fff; padding: 6px 10px; border-radius: 8px; font-weight: 700; }
.footer { display:flex; gap: 10px; margin-top: 8px; }
.link { background: transparent; border: none; color: #0b69ff; cursor: pointer; padding: 0; }
.json { background:#0b1020; color:#cde4ff; padding: 12px; border-radius: 10px; max-height: 260px; overflow:auto; }
.spinner { width: 16px; height: 16px; border: 2px solid #ddd; border-top-color: #ff8a00; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
</style>