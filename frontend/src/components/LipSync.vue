<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus, VideoCamera, FolderOpened, Refresh } from '@element-plus/icons-vue'

const textInput = ref('I am a text-driven talking head. No audio needed.')
const selectedImage = ref<File | null>(null)
const imagePreview = ref<string | null>(null)
const isGenerating = ref(false)
const videoUrl = ref<string | null>(null)
const errorMsg = ref<string | null>(null)
const projects = ref<any[]>([])
const showProjectsDialog = ref(false)
const selectedProjectVideo = ref<string | null>(null)

const API_BASE_URL = 'http://localhost:8000'

const handleImageChange = (uploadFile: any) => {
  const file = uploadFile.raw
  if (file) {
    selectedImage.value = file
    imagePreview.value = URL.createObjectURL(file)
    errorMsg.value = null
  }
}

const removeImage = () => {
  selectedImage.value = null
  imagePreview.value = null
}

const generateVideo = async () => {
  if (!textInput.value) {
    ElMessage.warning('Please enter some text.')
    return
  }
  if (!selectedImage.value) {
    ElMessage.warning('Please upload a reference image.')
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
      ElMessage.success('Video generated successfully!')
      fetchProjects()
    } else {
      const msg = data.detail || 'Generation failed.'
      errorMsg.value = msg
      ElMessage.error(msg)
    }
  } catch (err) {
    const msg = 'Error connecting to the backend. Is it running?'
    errorMsg.value = msg
    ElMessage.error(msg)
    console.error(err)
  } finally {
    isGenerating.value = false
  }
}

const fetchProjects = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/projects`)
    const data = await response.json()
    if (data.success) {
      projects.value = data.projects
    }
  } catch (err) {
    console.error('Failed to fetch projects:', err)
  }
}

const openProjectsDialog = () => {
  fetchProjects()
  showProjectsDialog.value = true
}

const playProjectVideo = (videoUrl: string) => {
  selectedProjectVideo.value = `${API_BASE_URL}${videoUrl}`
}

const formatDate = (dateStr: string) => {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  return d.toLocaleString()
}

const truncateText = (text: string, maxLen: number = 50) => {
  if (!text) return ''
  return text.length > maxLen ? text.substring(0, maxLen) + '...' : text
}

onMounted(() => {
  fetchProjects()
})
</script>

<template>
  <div class="lip-sync-container">
    <!-- Header -->
    <el-row justify="center" class="header-row">
      <el-col :span="22">
        <h1 class="app-title">
          <el-icon :size="32" color="#409eff"><VideoCamera /></el-icon>
          Text-to-LipSync Video Generator
        </h1>
        <p class="subtitle">Generate lip-synced videos from plain text and a single image</p>
      </el-col>
    </el-row>

    <!-- Main Content -->
    <el-row :gutter="24" justify="center">
      <el-col :xs="24" :sm="24" :md="12" :lg="10">
        <!-- Generate Card -->
        <el-card shadow="always" class="input-card">
          <template #header>
            <div class="card-header">
              <span>Generate New Video</span>
              <el-button type="primary" :icon="FolderOpened" text @click="openProjectsDialog">
                Browse Projects
              </el-button>
            </div>
          </template>

          <el-form label-position="top">
            <!-- Text Input -->
            <el-form-item label="1. Enter Text">
              <el-input
                v-model="textInput"
                type="textarea"
                :rows="4"
                placeholder="Type what you want the person to say..."
              />
            </el-form-item>

            <!-- Image Upload -->
            <el-form-item label="2. Upload Reference Image">
              <div v-if="!imagePreview" class="upload-wrapper">
                <el-upload
                  :auto-upload="false"
                  :show-file-list="false"
                  accept="image/*"
                  :on-change="handleImageChange"
                >
                  <div class="upload-placeholder">
                    <el-icon :size="40" color="#c0c4cc"><Plus /></el-icon>
                    <span>Click to upload face image</span>
                    <span class="upload-hint">Supports JPG, PNG</span>
                  </div>
                </el-upload>
              </div>
              <div v-else class="preview-wrapper">
                <el-image :src="imagePreview" fit="contain" class="preview-img" />
                <el-button type="danger" size="small" class="remove-btn" @click="removeImage">
                  Remove
                </el-button>
              </div>
            </el-form-item>

            <!-- Generate Button -->
            <el-form-item>
              <el-button
                type="primary"
                :loading="isGenerating"
                :disabled="isGenerating"
                class="generate-btn"
                @click="generateVideo"
              >
                <template v-if="isGenerating">
                  Generating... (May take a minute)
                </template>
                <template v-else>
                  Generate Lip-Sync Video
                </template>
              </el-button>
            </el-form-item>
          </el-form>

          <!-- Error Message -->
          <el-alert
            v-if="errorMsg"
            :title="errorMsg"
            type="error"
            show-icon
            :closable="false"
          />
        </el-card>
      </el-col>

      <el-col :xs="24" :sm="24" :md="12" :lg="10">
        <!-- Result Card -->
        <el-card shadow="always" class="result-card">
          <template #header>
            <div class="card-header">
              <span>3. Output Video</span>
            </div>
          </template>

          <div class="video-container">
            <!-- Loading State -->
            <div v-if="isGenerating" class="loading-state">
              <el-progress type="circle" :percentage="50" :stroke-width="6" status="warning" />
              <p class="loading-text">Diffusion model is processing...</p>
            </div>

            <!-- Video Result -->
            <div v-else-if="videoUrl" class="video-wrapper">
              <video :src="videoUrl" controls autoplay class="result-video"></video>
            </div>

            <!-- Empty State -->
            <div v-else class="empty-state">
              <el-empty description="Your video will appear here" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Projects Dialog -->
    <el-dialog
      v-model="showProjectsDialog"
      title="Generated Projects"
      width="80%"
      :top="'5vh'"
      class="projects-dialog"
    >
      <el-row :gutter="16">
        <!-- Project List -->
        <el-col :span="selectedProjectVideo ? 14 : 24">
          <el-table :data="projects" stripe style="width: 100%" max-height="500">
            <el-table-column prop="text" label="Text" min-width="200">
              <template #default="{ row }">
                <el-tooltip :content="row.text" placement="top">
                  <span>{{ truncateText(row.text) }}</span>
                </el-tooltip>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="Created At" width="180">
              <template #default="{ row }">
                {{ formatDate(row.created_at) }}
              </template>
            </el-table-column>
            <el-table-column label="Action" width="120" fixed="right">
              <template #default="{ row }">
                <el-button
                  type="primary"
                  size="small"
                  :icon="VideoCamera"
                  @click="playProjectVideo(row.video_url)"
                >
                  Play
                </el-button>
              </template>
            </el-table-column>
          </el-table>

          <div v-if="projects.length === 0" class="empty-projects">
            <el-empty description="No projects generated yet" />
          </div>
        </el-col>

        <!-- Video Preview -->
        <el-col :span="10" v-if="selectedProjectVideo">
          <el-card shadow="never">
            <template #header>
              <span>Video Preview</span>
            </template>
            <video :src="selectedProjectVideo" controls class="preview-video"></video>
          </el-card>
        </el-col>
      </el-row>

      <template #footer>
        <el-button @click="showProjectsDialog = false">Close</el-button>
        <el-button type="primary" :icon="Refresh" @click="fetchProjects">Refresh</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.lip-sync-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Inter', -apple-system, sans-serif;
}

.header-row {
  margin-bottom: 32px;
  text-align: center;
}

.app-title {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  font-size: 28px;
  margin: 0 0 8px 0;
  color: #303133;
}

.subtitle {
  color: #909399;
  font-size: 16px;
  margin: 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 16px;
}

.input-card,
.result-card {
  margin-bottom: 24px;
  border-radius: 12px;
}

.upload-wrapper {
  border: 2px dashed #dcdfe6;
  border-radius: 8px;
  padding: 32px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
  background: #fafafa;
}

.upload-wrapper:hover {
  border-color: #409eff;
  background: #ecf5ff;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: #909399;
}

.upload-hint {
  font-size: 12px;
  color: #c0c4cc;
}

.preview-wrapper {
  position: relative;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  max-height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #fafafa;
}

.preview-img {
  max-width: 100%;
  max-height: 250px;
  object-fit: contain;
}

.remove-btn {
  position: absolute;
  top: 8px;
  right: 8px;
}

.generate-btn {
  width: 100%;
  font-size: 16px;
  padding: 22px 0;
  border-radius: 8px;
}

.video-container {
  min-height: 350px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
}

.loading-state {
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.loading-text {
  color: #909399;
  margin: 0;
}

.video-wrapper {
  width: 100%;
}

.result-video {
  width: 100%;
  max-height: 500px;
  display: block;
}

.empty-state {
  padding: 40px 0;
}

.projects-dialog :deep(.el-dialog__body) {
  padding-top: 20px;
}

.empty-projects {
  padding: 40px 0;
}

.preview-video {
  width: 100%;
  max-height: 400px;
  border-radius: 4px;
}
</style>
