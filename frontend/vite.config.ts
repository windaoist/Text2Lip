import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/generate': { target: 'http://localhost:8000', changeOrigin: true },
      '/projects': { target: 'http://localhost:8000', changeOrigin: true },
      '/outputs': { target: 'http://localhost:8000', changeOrigin: true },
      '/generate-stream': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
