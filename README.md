# Text2Lip — 纯文本驱动的唇动视频生成

> 本项目实现了 **文本 → 视素(Viseme) → 唇动视频** 的端到端纯文本驱动流水线，无需任何音频输入。

---

## 项目核心架构

整个流水线遵循 **1→2→3 三步架构**：

### Step 1: 文本 → 视素 (Text → Viseme)

| 模块 | 文件 | 核心技术 |
|------|------|----------|
| 音素解析 | `lip_sync/models/text_to_viseme.py` | **g2p-en** 将英文文本转为音素序列（IPA） |
| 视素映射 | 同上 | 基于语言学规则的 **音素→视素映射表**（16个视素类别） |
| 时长预测 | 同上（`process_with_duration`） | 基于语言统计学为每个音素分配帧时长（如短元音2-3帧、长元音4-5帧） |

**输入**: 纯文本（如 "I am a text-driven talking head."）  
**输出**: 视素 ID 序列 + 每音素帧数 + 总帧数

### Step 2: 视素 → 运动特征 (Viseme → Motion Features)

| 模块 | 文件 | 核心技术 |
|------|------|----------|
| 视素编码器 | `lip_sync/models/motion_gen.py` | **Transformer Encoder**（4层自注意力）+ **Cross-Attention Decoder**（2层） |
| 帧对齐 | 同上 | 可学习的帧位置嵌入，将 N 个视素动态对齐到 T 个输出帧 |
| 微表情条件 | 同上（可选） | **FAU**（面部动作单元）信号注入，用于微表情控制 |

**输入**: (B, N) 视素 ID 序列 + target_frames_len  
**输出**: (B, T, 19200) 运动特征向量（与 Whisper 特征维度一致）

### Step 3: 运动特征 → 视频 (Features → Video)

| 模块 | 文件 | 核心技术 |
|------|------|----------|
| EchoMimic 后端 | `lip_sync/inference/echomimic_backend.py` | **Stable Diffusion + 3D UNet + Cross-Attention** |
| 人脸检测 | 同上 | **MTCNN** 人脸检测 + **MediaPipe FaceMesh** 关键点 |
| 视频生成 | 同上 | **DDIM Scheduler** 去噪 + 时序注意力机制 |

**输入**: 参考人脸图像 + (1, T, 50, 384) 特征张量  
**输出**: `.mp4` 唇动视频文件

### 辅助模块

| 模块 | 文件 | 用途 |
|------|------|------|
| 辅助渲染器 | `lip_sync/models/aux_renderer.py` | 训练阶段：将 19200 维特征解码为 64×64 低分辨率帧，用于计算 LPIPS 感知损失 |
| SyncNet | `lip_sync/models/syncnet.py` | 训练阶段：基于预训练 Wav2Lip 的 SyncNet 计算音画同步对比损失 |
| 数据预处理 | `lip_sync/data_utils/preprocess_grid.py` | GRID 数据集的多进程预处理（Whisper 特征提取、人脸检测、FAU 提取） |
| 训练脚本 | `lip_sync/train/train_text_to_feature.py` | 端到端训练：MSE损失 + LPIPS感知损失 + SyncNet同步损失的联合优化 |

---

## 项目结构

```
E:\playground\
├── lip_sync/                    # 核心代码
│   ├── inference_pipeline.py    # 端到端推理流水线入口
│   ├── models/
│   │   ├── text_to_viseme.py    # Step 1: 文本→视素（G2P + 音素时长预测）
│   │   ├── motion_gen.py        # Step 2: 视素→运动特征（Transformer编码+交叉注意力解码）
│   │   ├── aux_renderer.py      # 辅助解码器（训练用，将特征渲染为低分辨率帧）
│   │   ├── syncnet.py           # SyncNet 同步损失计算（训练用）
│   ├── train/
│   │   └── train_text_to_feature.py  # 训练脚本（MSE+LPIPS+SyncNet联合损失）
│   ├── inference/
│   │   └── echomimic_backend.py  # Step 3: EchoMimic 视频生成后端
│   └── data_utils/
│       └── preprocess_grid.py   # GRID 数据集预处理（多进程加速）
├── configs/
│   ├── text_driven_config.yaml  # 训练/推理配置文件
│   └── inference/               # EchoMimic 推理配置
├── backend_server.py            # FastAPI 后端服务
├── frontend/                    # Vue 3 + Element Plus 前端
│   └── src/components/LipSync.vue
├── train_and_gen.ipynb          # Colab 训练与生成笔记
├── third_party/EchoMimic/       # EchoMimic 第三方依赖
└── scripts/download_models.py   # 模型下载脚本
```

---

## 环境配置

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载预训练模型
python scripts/download_models.py

# 3. 启动后端
python backend_server.py

# 4. 启动前端（新终端）
cd frontend && npm install && npm run dev
```

### Colab 训练

参见 `train_and_gen.ipynb`，该笔记包含了完整的：
- 环境配置（注意保留 `protobuf`/`mediapipe` 兼容性修复命令）
- GRID 数据集下载与预处理
- 模型训练
- 推理演示

---

## 数据集

训练使用 **GRID 数据集**（speaker s1 子集）：
- 视频: [s1.mpg_vcd.zip](https://spandh.dcs.shef.ac.uk//gridcorpus/s1/video/s1.mpg_vcd.zip)
- 对齐文件: [s1.tar](https://spandh.dcs.shef.ac.uk/gridcorpus/s1/align/s1.tar)
- 音频: [s1.tar](https://spandh.dcs.shef.ac.uk/gridcorpus/s1/audio/s1.tar)

预处理命令:
```bash
python lip_sync/data_utils/preprocess_grid.py --limit 1000 --workers 1 --skip 3
```

训练命令:
```bash
python lip_sync/train/train_text_to_feature.py
```

---

## 关键依赖

| 依赖 | 用途 | 注意 |
|------|------|------|
| PyTorch ≥ 2.2 | 深度学习框架 | 支持 torch.compile 加速 |
| diffusers 0.24.0 | Stable Diffusion 管线 | EchoMimic 依赖 |
| transformers 4.38.1 | HuggingFace 模型加载 | EchoMimic 依赖 |
| EchoMimic (third_party) | 扩散模型视频生成 | 需要 SD 和 EchoMimic 权重 |
| g2p-en | 英文文本→音素 | Step 1 核心 |
| mediapipe 0.10.13 | 面部关键点检测 | FAU 提取 |
| facenet-pytorch (MTCNN) | 人脸检测 | 参考图像预处理 |

---

## 技术亮点

1. **纯文本驱动**: 完全绕过 TTS+ASR 管线，从文本直接生成特征
2. **语言学时长预测**: 基于音素类别的统计时长模型，替代固定帧分配
3. **Cross-Attention 对齐**: 用 Transformer Decoder 将短视素序列动态对齐到长帧序列
4. **多目标联合训练**: MSE + LPIPS 感知损失 + SyncNet 同步损失的加权优化
5. **EchoMimic 集成**: 利用预训练的扩散模型将特征转化为高质量视频
