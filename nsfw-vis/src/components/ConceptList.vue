<template>
  <div class="panel">
    <ul class="concepts">
      <li v-for="(v, k) in sorted" :key="k">
        <span class="cname">{{ k }}:</span>
        <span class="cval">{{ fmt(v) }}</span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{ concepts: Record<string, number> }>();
const fmt = (x: number) => x.toFixed(2);
const sorted = computed(() => Object.fromEntries(Object.entries(props.concepts || {}).sort((a,b)=>b[1]-a[1])));
</script>

<style scoped>
.panel { border:2px solid #bdbdbd; border-radius: 10px; padding: 14px; }
.concepts { list-style: none; padding:0; margin:0; }
.concepts li { display:flex; justify-content: space-between; gap: 14px; font-size: 15px; padding: 6px 0; border-bottom: 1px dashed #ddd; }
.cname { color:#333; font-weight:600; }
.cval { color:#444; }
</style>