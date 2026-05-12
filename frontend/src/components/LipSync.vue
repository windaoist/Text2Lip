<script setup lang="ts">
import { ref } from 'vue'

const textInput = ref('I am a text-driven talking head. No audio needed.')
const selectedImage = ref<File | null>(null)
const imagePreview = ref<string | null>(null)
const isGenerating = ref(false)
const videoUrl = ref<string | null>(null)
const errorMsg = ref<string | null>(null)

const API_BASE_URL = 'http://localhost:8000'

const handleImageChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files[0]) {
    const file = target.files[0]
    selectedImage.ref = file // Wait, ref usage in script setup is different
    // corrected:
    selectedImage.value = file
    imagePreview.value = URL.createObjectURL(file)
    errorMsg.value = null
  }
}

const generateVideo = async () => {
  if (!textInput.value) {
    errorMsg.value = 'Please enter some text.'
    return
  }
  if (!selectedImage.value) {
    errorMsg.value = 'Please upload a reference image.'
    return
  }

  isGenerating.value = true
  errorMsg.value = null
  videoUrl.value = null

  const formData = new FormData()
  formData.append('text', textInput.value)
  formData.append('image', selectedImage.value)

  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()
    if (data.success) {
      videoUrl.value = `${API_BASE_URL}${data.video_url}`
    } else {
      errorMsg.value = data.detail || 'Generation failed.'
    }
  } catch (err) {
    errorMsg.value = 'Error connecting to the backend. Is it running?'
    console.error(err)
  } finally {
    isGenerating.value = false
  }
}
</script>

<template>
  <div class="lip-sync-container">
    <h1>Text-to-LipSync Video Generator</h1>
    <p class="subtitle">Generate lip-synced videos from plain text and a single image.</p>

    <div class="main-layout">
      <!-- Input Section -->
      <div class="input-panel">
        <div class="form-group">
          <label>1. Enter Text</label>
          <textarea 
            v-model="textInput" 
            placeholder="Type what you want the person to say..."
            rows="4"
          ></textarea>
        </div>

        <div class="form-group">
          <label>2. Upload Reference Image</label>
          <div class="upload-area" :class="{ 'has-image': imagePreview }">
            <input type="file" @change="handleImageChange" accept="image/*" id="file-input" hidden />
            <label for="file-input" class="upload-label">
              <div v-if="!imagePreview" class="upload-placeholder">
                <span class="icon">📷</span>
                <span>Click to upload face image</span>
              </div>
              <img v-else :src="imagePreview" class="preview-img" />
            </label>
          </div>
        </div>

        <button 
          @click="generateVideo" 
          :disabled="isGenerating"
          class="generate-btn"
        >
          <span v-if="isGenerating">⚡ Generating... (May take a minute)</span>
          <span v-else>🚀 Generate Lip-Sync Video</span>
        </button>

        <div v-if="errorMsg" class="error-banner">
          {{ errorMsg }}
        </div>
      </div>

      <!-- Result Section -->
      <div class="result-panel">
        <label>3. Output Video</label>
        <div class="video-container">
          <div v-if="isGenerating" class="loading-state">
            <div class="spinner"></div>
            <p>Diffusion model is processing...</p>
          </div>
          <video v-else-if="videoUrl" :src="videoUrl" controls autoplay class="result-video"></video>
          <div v-else class="empty-state">
            <p>Your video will appear here</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.lip-sync-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Inter', -apple-system, sans-serif;
}

h1 {
  text-align: center;
  margin-bottom: 8px;
  color: #2c3e50;
}

.subtitle {
  text-align: center;
  color: #666;
  margin-bottom: 40px;
}

.main-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 40px;
}

@media (max-width: 768px) {
  .main-layout {
    grid-template-columns: 1fr;
  }
}

.form-group {
  margin-bottom: 24px;
}

label {
  display: block;
  font-weight: 600;
  margin-bottom: 10px;
  color: #444;
}

textarea {
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 16px;
  resize: vertical;
  transition: border-color 0.3s;
}

textarea:focus {
  outline: none;
  border-color: #42b883;
}

.upload-area {
  border: 2px dashed #ccc;
  border-radius: 8px;
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  overflow: hidden;
  transition: all 0.3s;
}

.upload-area:hover {
  border-color: #42b883;
  background: #f9fffb;
}

.upload-area.has-image {
  border-style: solid;
}

.upload-label {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  color: #888;
}

.icon {
  font-size: 32px;
  margin-bottom: 8px;
}

.preview-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.generate-btn {
  width: 100%;
  padding: 16px;
  background: #42b883;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s, transform 0.1s;
}

.generate-btn:hover {
  background: #3aa876;
}

.generate-btn:active {
  transform: scale(0.98);
}

.generate-btn:disabled {
  background: #a8d5c0;
  cursor: not-allowed;
}

.error-banner {
  margin-top: 16px;
  padding: 12px;
  background: #fff0f0;
  color: #d63031;
  border-radius: 6px;
  border-left: 4px solid #d63031;
}

.video-container {
  background: #f8f9fa;
  border-radius: 12px;
  border: 2px solid #e9ecef;
  aspect-ratio: 1 / 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.result-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.loading-state {
  text-align: center;
  padding: 20px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #42b883;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.empty-state {
  color: #adb5bd;
}
</style>
