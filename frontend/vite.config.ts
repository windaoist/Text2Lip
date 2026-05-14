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
      '/generate-ws': { target: 'http://localhost:8000', changeOrigin: true },
      // WebSocket 代理: 前端 ws://host:5173/ws/xxx → 后端 ws://localhost:8000/ws/xxx (中文注释)
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
