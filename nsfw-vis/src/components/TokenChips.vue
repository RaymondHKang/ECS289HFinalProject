<template>
  <div class="chips">
    <span class="label">Output:</span>
    <span v-for="(tok, i) in tokens" :key="i" class="tok" :style="chipStyle(scores[i])" :title="'importance: ' + fmt(scores[i])">
      {{ tok }}
    </span>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{ tokens: string[]; scores: number[] }>();
const fmt = (x: number) => x.toFixed(2);
const chipStyle = (s: number) => ({
  background: `rgba(255, 136, 0, ${Math.min(Math.max(s,0),1)*0.35})`,
  borderColor: `rgba(255, 136, 0, ${Math.min(Math.max(s,0),1)*0.6})`
});
</script>

<style scoped>
.chips { border: 4px solid #8c8c8c; border-radius: 12px; padding: 10px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.label { font-weight: 700; margin-right: 6px; }
.tok { display:inline-block; padding: 2px 6px; border:1px solid transparent; border-radius: 8px; }
</style>