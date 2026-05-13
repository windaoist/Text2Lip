<script setup lang="ts">
import LipSync from './components/LipSync.vue'
</script>

<template>
  <div class="app-wrapper">
    <LipSync />
  </div>
</template>

<style>
body {
  margin: 0;
  background-color: #f5f7fa;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.app-wrapper {
  min-height: 100vh;
}
</style>
