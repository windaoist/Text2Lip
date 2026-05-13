<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus, VideoCamera, FolderOpened, Refresh, ZoomIn, ZoomOut, FullScreen } from '@element-plus/icons-vue'

const textInput = ref('I am a text-driven talking head. No audio needed.')
const selectedImage = ref<File | null>(null)
const imagePreview = ref<string | null>(null)
const isGenerating = ref(false)
const videoUrl = ref<string | null>(null)
const errorMsg = ref<string | null>(null)
const projects = ref<any[]>([])
const showProjectsDialog = ref(false)
const selectedProjectVideo = ref<string | null>(null)

// ===== Image Zoom State =====
const imageScale = ref(1)
const imageNaturalWidth = ref(0)
const imageNaturalHeight = ref(0)
const IMAGE_MIN_SCALE = 0.1
const IMAGE_MAX_SCALE = 5
const IMAGE_ZOOM_STEP = 0.25

// ===== Progress State =====
const progressPercent = ref(0)
const progressStage = ref('')
const progressMessage = ref('')
const progressStages = ref<{ stage: string; label: string; percent: number }[]>([
  { stage: 'start', label: '开始生成', percent: 0 },
  { stage: 'text_to_viseme', label: '文本转口型序列', percent: 0 },
  { stage: 'initializing_backend', label: '初始化引擎', percent: 5 },
  { stage: 'preprocess', label: '预处理图像', percent: 0 },
  { stage: 'text_features', label: '文本特征提取', percent: 5 },
  { stage: 'diffusion_start', label: '扩散生成准备', percent: 10 },
  { stage: 'diffusion', label: '扩散推理', percent: 40 },
  { stage: 'frame_generation', label: '逐帧生成', percent: 70 },
  { stage: 'saving', label: '保存视频', percent: 95 },
  { stage: 'complete', label: '生成完成', percent: 100 },
])
const currentStageDetail = computed(() => {
  const found = progressStages.value.find(s => s.stage === progressStage.value)
  return found ? found.label : progressStage.value
})

// SSE handling
let eventSource: EventSource | null = null

const API_BASE_URL = ''

const handleImageChange = (uploadFile: any) => {
  const file = uploadFile.raw
  if (file) {
    selectedImage.value = file
    const url = URL.createObjectURL(file)
    // Revoke old URL if exists
    if (imagePreview.value) {
      URL.revokeObjectURL(imagePreview.value)
    }
    imagePreview.value = url
    imageScale.value = 1
    errorMsg.value = null

    // Get natural image dimensions
    const img = new Image()
    img.onload = () => {
      imageNaturalWidth.value = img.naturalWidth
      imageNaturalHeight.value = img.naturalHeight
    }
    img.src = url
  }
}

const removeImage = () => {
  selectedImage.value = null
  if (imagePreview.value) {
    URL.revokeObjectURL(imagePreview.value)
  }
  imagePreview.value = null
  imageScale.value = 1
  imageNaturalWidth.value = 0
  imageNaturalHeight.value = 0
}

// ===== Zoom Controls =====
const zoomIn = () => {
  imageScale.value = Math.min(imageScale.value + IMAGE_ZOOM_STEP, IMAGE_MAX_SCALE)
}

const zoomOut = () => {
  imageScale.value = Math.max(imageScale.value - IMAGE_ZOOM_STEP, IMAGE_MIN_SCALE)
}

const zoomReset = () => {
  imageScale.value = 1
}

const handleWheel = (event: WheelEvent) => {
  if (!imagePreview.value) return
  event.preventDefault()
  const delta = event.deltaY > 0 ? -IMAGE_ZOOM_STEP : IMAGE_ZOOM_STEP
  imageScale.value = Math.min(Math.max(imageScale.value + delta, IMAGE_MIN_SCALE), IMAGE_MAX_SCALE)
}

// ===== SSE Progress Helpers =====
const resetProgress = () => {
  progressPercent.value = 0
  progressStage.value = ''
  progressMessage.value = ''
}

const updateProgressFromSSE = (data: any) => {
  if (data.type === 'progress') {
    progressPercent.value = Math.round(data.percent)
    progressStage.value = data.stage
    progressMessage.value = data.message
  } else if (data.type === 'complete') {
    progressPercent.value = 100
    progressStage.value = 'complete'
    progressMessage.value = '生成完成!'
    videoUrl.value = `${API_BASE_URL}${data.video_url}`
    ElMessage.success('视频生成成功!')
    fetchProjects()
  } else if (data.type === 'error') {
    errorMsg.value = data.message
    ElMessage.error(data.message)
    isGenerating.value = false
    resetProgress()
  } else if (data.type === 'done') {
    isGenerating.value = false
  }
}

const generateVideo = async () => {
  if (!textInput.value) {
    ElMessage.warning('请输入文本内容.')
    return
  }
  if (!selectedImage.value) {
    ElMessage.warning('请上传参考照片.')
    return
  }

  isGenerating.value = true
  errorMsg.value = null
  videoUrl.value = null
  resetProgress()

  const formData = new FormData()
  formData.append('text', textInput.value)
  formData.append('image', selectedImage.value)

  try {
    // Use SSE streaming endpoint
    const response = await fetch(`${API_BASE_URL}/generate-stream`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('No response body reader available')
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      
      // Parse SSE events
      const lines = buffer.split('\n')
      buffer = '' // reset, we'll reconstruct
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6)
          try {
            const data = JSON.parse(jsonStr)
            updateProgressFromSSE(data)
          } catch (e) {
            console.warn('Failed to parse SSE data:', jsonStr)
          }
        } else if (line !== '') {
          // Keep non-empty non-data lines in buffer (might be part of next event)
          buffer += line + '\n'
        }
      }
    }

    // Handle any remaining buffer
    if (buffer.trim()) {
      const lines = buffer.split('\n')
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            updateProgressFromSSE(data)
          } catch (e) {
            // ignore
          }
        }
      }
    }

  } catch (err) {
    const msg = '连接后端出错，请确认后端是否运行中.'
    errorMsg.value = msg
    ElMessage.error(msg)
    console.error(err)
  } finally {
    isGenerating.value = false
  }
}

// Fallback to old API if SSE fails
const generateVideoFallback = async () => {
  const formData = new FormData()
  formData.append('text', textInput.value)
  formData.append('image', selectedImage.value!)

  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()
    if (data.success) {
      videoUrl.value = `${API_BASE_URL}${data.video_url}`
      ElMessage.success('视频生成成功!')
      fetchProjects()
    } else {
      const msg = data.detail || '生成失败.'
      errorMsg.value = msg
      ElMessage.error(msg)
    }
  } catch (err) {
    const msg = '连接后端出错，请确认后端是否运行中.'
    errorMsg.value = msg
    ElMessage.error(msg)
    console.error(err)
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
    console.error('获取项目列表失败:', err)
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

onUnmounted(() => {
  // Cleanup
  if (imagePreview.value) {
    URL.revokeObjectURL(imagePreview.value)
  }
})
</script>

<template>
  <div class="lip-sync-container">
    <!-- Header -->
    <el-row justify="center" class="header-row">
      <el-col :span="22">
        <h1 class="app-title">
          <el-icon :size="32" color="#409eff"><VideoCamera /></el-icon>
          文字驱动唇形同步视频生成器
        </h1>
        <p class="subtitle">只需一段文字和一张照片，即可生成唇形同步视频</p>
      </el-col>
    </el-row>

    <!-- Main Content -->
    <el-row :gutter="24" justify="center">
      <el-col :xs="24" :sm="24" :md="12" :lg="10">
        <!-- Generate Card -->
        <el-card shadow="always" class="input-card">
          <template #header>
            <div class="card-header">
              <span>生成新视频</span>
              <el-button type="primary" :icon="FolderOpened" text @click="openProjectsDialog">
                浏览历史记录
              </el-button>
            </div>
          </template>

          <el-form label-position="top">
            <!-- Text Input -->
            <el-form-item label="1. 输入文本">
              <el-input
                v-model="textInput"
                type="textarea"
                :rows="4"
                placeholder="输入你想让视频人物说的话..."
              />
            </el-form-item>

            <!-- Image Upload with Zoom -->
            <el-form-item label="2. 上传参考照片">
              <div v-if="!imagePreview" class="upload-wrapper">
                <el-upload
                  :auto-upload="false"
                  :show-file-list="false"
                  accept="image/*"
                  :on-change="handleImageChange"
                >
                  <div class="upload-placeholder">
                    <el-icon :size="40" color="#c0c4cc"><Plus /></el-icon>
                    <span>点击上传人脸照片</span>
                    <span class="upload-hint">支持 JPG、PNG 格式</span>
                  </div>
                </el-upload>
              </div>
              <div v-else class="preview-wrapper">
                <!-- Zoom Toolbar -->
                <div class="zoom-toolbar">
                  <span class="zoom-label">缩放: {{ Math.round(imageScale * 100) }}%</span>
                  <div class="zoom-actions">
                    <el-button size="small" circle :icon="ZoomOut" @click="zoomOut" :disabled="imageScale <= IMAGE_MIN_SCALE" />
                    <el-button size="small" circle :icon="FullScreen" @click="zoomReset" :disabled="imageScale === 1" />
                    <el-button size="small" circle :icon="ZoomIn" @click="zoomIn" :disabled="imageScale >= IMAGE_MAX_SCALE" />
                  </div>
                </div>
                <!-- Image Preview Container with Scroll -->
                <div class="preview-scroll-container" @wheel.prevent="handleWheel">
                  <div class="preview-inner" :style="{ transform: `scale(${imageScale})` }">
                    <el-image :src="imagePreview" fit="contain" class="preview-img" />
                  </div>
                </div>
                <!-- Image Info & Remove -->
                <div class="preview-footer">
                  <span class="image-info" v-if="imageNaturalWidth">
                    {{ imageNaturalWidth }} × {{ imageNaturalHeight }}px
                  </span>
                  <el-button type="danger" size="small" class="remove-btn" @click="removeImage">
                    移除
                  </el-button>
                </div>
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
                  {{ progressMessage || '正在生成...（可能需要一分钟）' }}
                </template>
                <template v-else>
                  生成唇形同步视频
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
              <span>3. 输出视频</span>
            </div>
          </template>

          <div class="video-container">
            <!-- Loading State with Real-time Progress -->
            <div v-if="isGenerating" class="loading-state">
              <el-progress 
                type="circle" 
                :percentage="Math.round(progressPercent)" 
                :stroke-width="6" 
                :status="progressPercent >= 100 ? 'success' : 'warning'"
              >
                <span class="progress-percentage">{{ Math.round(progressPercent) }}%</span>
              </el-progress>
              <div class="progress-details">
                <p class="loading-text">
                  <span class="stage-indicator">{{ currentStageDetail }}</span>
                  <span v-if="progressMessage && progressMessage !== currentStageDetail" class="stage-message"> — {{ progressMessage }}</span>
                </p>
                <!-- Mini progress bar for stages context -->
                <div class="stage-progress-bar">
                  <div 
                    v-for="(stage, idx) in progressStages" 
                    :key="stage.stage"
                    class="stage-segment"
                    :class="{ 
                      'completed': progressPercent >= stage.percent && stage.percent > 0,
                      'current': progressStage === stage.stage,
                      'pending': progressPercent < stage.percent
                    }"
                    :style="{ 
                      flex: progressStages.length === 7 ? 1 : 'auto',
                      minWidth: '8px'
                    }"
                  >
                    <div class="segment-inner"></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Video Result -->
            <div v-else-if="videoUrl" class="video-wrapper">
              <video :src="videoUrl" controls autoplay class="result-video"></video>
            </div>

            <!-- Empty State -->
            <div v-else class="empty-state">
              <el-empty description="生成的视频将显示在这里" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- Projects Dialog -->
    <el-dialog
      v-model="showProjectsDialog"
      title="已生成的项目"
      width="80%"
      :top="'5vh'"
      class="projects-dialog"
    >
      <el-row :gutter="16">
        <!-- Project List -->
        <el-col :span="selectedProjectVideo ? 14 : 24">
          <el-table :data="projects" stripe style="width: 100%" max-height="500">
            <el-table-column prop="text" label="文本" min-width="200">
              <template #default="{ row }">
                <el-tooltip :content="row.text" placement="top">
                  <span>{{ truncateText(row.text) }}</span>
                </el-tooltip>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="创建时间" width="180">
              <template #default="{ row }">
                {{ formatDate(row.created_at) }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="120" fixed="right">
              <template #default="{ row }">
                <el-button
                  type="primary"
                  size="small"
                  :icon="VideoCamera"
                  @click="playProjectVideo(row.video_url)"
                >
                  播放
                </el-button>
              </template>
            </el-table-column>
          </el-table>

          <div v-if="projects.length === 0" class="empty-projects">
            <el-empty description="暂无已生成的项目" />
          </div>
        </el-col>

        <!-- Video Preview -->
        <el-col :span="10" v-if="selectedProjectVideo">
          <el-card shadow="never">
            <template #header>
              <span>视频预览</span>
            </template>
            <video :src="selectedProjectVideo" controls class="preview-video"></video>
          </el-card>
        </el-col>
      </el-row>

      <template #footer>
        <el-button @click="showProjectsDialog = false">关闭</el-button>
        <el-button type="primary" :icon="Refresh" @click="fetchProjects">刷新</el-button>
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

/* ===== Upload Styles ===== */
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

/* ===== Preview with Zoom ===== */
.preview-wrapper {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  background: #fafafa;
}

.zoom-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
}

.zoom-label {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
}

.zoom-actions {
  display: flex;
  gap: 4px;
}

.preview-scroll-container {
  width: 100%;
  max-height: 350px;
  overflow: auto;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  cursor: grab;
}

.preview-scroll-container:active {
  cursor: grabbing;
}

.preview-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  transform-origin: center center;
  transition: transform 0.15s ease;
}

.preview-img {
  max-width: 100%;
  display: block;
}

.preview-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: #f5f7fa;
  border-top: 1px solid #e4e7ed;
}

.image-info {
  font-size: 12px;
  color: #909399;
}

.remove-btn {
}

.generate-btn {
  width: 100%;
  font-size: 16px;
  padding: 22px 0;
  border-radius: 8px;
}

/* ===== Video Container ===== */
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
  padding: 32px 16px;
}

.progress-percentage {
  font-size: 22px;
  font-weight: 700;
  color: #e6a23c;
}

.progress-details {
  width: 100%;
  max-width: 280px;
}

.loading-text {
  color: #606266;
  margin: 0 0 12px 0;
  font-size: 14px;
}

.stage-indicator {
  font-weight: 600;
  color: #409eff;
}

.stage-message {
  color: #909399;
  font-size: 13px;
}

.stage-progress-bar {
  display: flex;
  gap: 2px;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  background: #ebeef5;
}

.stage-segment {
  flex: 1;
  transition: all 0.3s ease;
}

.segment-inner {
  height: 100%;
  border-radius: 3px;
  transition: background-color 0.3s ease, opacity 0.3s ease;
}

.stage-segment.completed .segment-inner {
  background-color: #67c23a;
}

.stage-segment.current .segment-inner {
  background-color: #409eff;
  animation: pulse 1.5s ease-in-out infinite;
}

.stage-segment.pending .segment-inner {
  background-color: #ebeef5;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
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

/* ===== Projects Dialog ===== */
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
