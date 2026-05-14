# Text2Lip — 纯文本驱动的唇动视频生成

> 本项目实现了 **文本 → 视素(Viseme) → 唇动视频** 的端到端纯文本驱动流水线，无需任何音频输入。

---

## 项目核心架构

整个流水线遵循 **1→2→3 三步架构**，所有核心代码位于 `lip_sync/` 目录下：

### Step 1: 文本 → 视素 (Text → Viseme)

**文件**: `lip_sync/models/text_to_viseme.py`

**类**: `TextToVisemeProcessor`

**实现细节**:

1. **G2P 音素转换**: 使用 `g2p-en` 库将英文文本转为 ARPAbet 音素序列。示例：`"I am"` → `['AY1', ' ', 'AE1', 'M']`
2. **音素→视素映射表**: 预定义了16个视素类别的映射规则，覆盖所有英语音素：
   - 视素0（静止/停顿）: `SIL`, `SP` 及所有标点
   - 视素1（AA/AO/AH）: 后低/中元音
   - 视素2（AE/EH/IH/IY/EY）: 前元音
   - 视素3（AW/OW/UW/UH/AY/OY）: 圆唇元音和双元音
   - 视素4（ER）: 卷舌元音
   - 视素5（P/B/M）: 双唇音
   - 视素6（F/V）: 唇齿音
   - 视素7（T/D/S/Z）: 齿龈音
   - 视素8（TH/DH）: 齿间音
   - 视素9（SH/ZH/CH/JH）: 腭龈音/塞擦音
   - 视素10（K/G/N/NG）: 软腭音
   - 视素11（L）: 边音
   - 视素12（R）: 卷舌近音
   - 视素13（Y）: 硬腭近音
   - 视素14（W）: 唇软腭近音
   - 视素15（HH）: 声门擦音
3. **语言学时长预测** (`process_with_duration`): 为每个音素分配基于语言学统计的帧数（1帧=40ms @25fps）：
   - 短元音（IH/EH/AE/AH/UH）: 3帧（120ms）
   - 长元音（IY/AA/AO/ER/OW/UW）: 5帧（200ms）
   - 双元音（AW/AY/EY/OY）: 6帧（240ms）
   - 擦音（F/V/TH/DH/S/Z/SH/ZH/HH）: 3帧（120ms）
   - 爆破音（P/B/T/D/K/G）: 2帧（80ms）
   - 塞擦音/鼻音/流音（CH/JH/M/N/NG/L/R）: 3帧（120ms）
   - 滑音（Y/W）: 2帧（80ms）
   - 停顿（空格/标点）: 2-4帧
4. **两种输出模式**:
   - `process()`: 返回未展开的视素ID序列，长度=N（音素数），每个音素1个token（训练时模型输入）
   - `process_with_duration()`: 返回展开后的视素序列+总帧数+每音素帧数+原始音素（用于计算目标视频时长）

**输入**: 纯文本字符串（如 `"I am a text-driven talking head. No audio needed."`）  
**输出**: 
- `viseme_ids`: `torch.LongTensor` 形状 `(N,)` — 短视素序列（训练/模型输入）
- `total_frames`: `int` — 展开后的目标视频帧数（用于控制视频长度）

---

### Step 2: 视素 → 运动特征 (Viseme → Motion Features)

**文件**: `lip_sync/models/motion_gen.py`

**类**: `VisemeEncoder`

**实现细节**:

1. **视素嵌入与位置编码**:
   - `nn.Embedding(num_visemes=16, d_model=512)` 将离散视素ID映射为连续向量
   - 正弦位置编码（`PositionalEncoding`）为序列注入顺序信息

2. **Transformer Encoder（自注意力编码器）**:
   - 4层 `TransformerEncoderLayer`，每层8头注意力
   - 将 N 个视素token编码为上下文感知的 memory 表示 `(B, N, 512)`
   - 支持 padding mask 处理批次中不同长度的视素序列

3. **Cross-Attention Decoder（交叉注意力解码器）**:
   - 2层 `TransformerDecoderLayer`
   - 可学习的帧位置查询嵌入 `frame_query_embed`（最大5000帧），将 T 个查询位置映射为 T 个帧特征
   - 通过 cross-attention 机制，Query=目标帧位置，Key/Value=视素编码memory
   - 实现从 N 个视素到 T 个输出帧的动态自适应对齐

4. **FAU 微表情条件注入**（可选）:
   - `fau_projection: Linear(16, 512)` 将16维面部动作单元信号投影到 d_model 空间
   - 训练时注入真实FAU信号，推理时使用全零信号
   - 为未来微表情控制提供接口

5. **输出投影**: `Linear(512, 19200)` 将帧特征映射到 19200 维（= 50 × 384，与 EchoMimic Whisper 特征维度完全一致）

**输入**: 
- `viseme_ids`: `(B, N)` 视素ID序列
- `target_frames_len`: `int` 目标输出帧数 T

**输出**: `(B, T, 19200)` 运动特征向量

---

### Step 3: 运动特征 → 视频 (Features → Video)

**文件**: `lip_sync/inference/echomimic_backend.py`

**类**: `EchoMimicBackend`

**实现细节**:

1. **模型初始化** (`initialize_models`):
   - **VAE** (AutoencoderKL): Stable Diffusion 的图像编解码器，负责潜在空间 ↔ 像素空间的转换
   - **Reference UNet** (UNet2DConditionModel): 2D UNet，从参考人脸图像中提取身份特征
   - **Denoising UNet** (EchoUNet3DConditionModel): 3D UNet，支持时序注意力，是核心的去噪网络
   - **Face Locator**: 轻量 CNN，将人脸mask编码为条件信号注入去噪过程
   - **Audio Processor** (Whisper): 音频特征提取器（纯文本模式下不实际使用，但预留了加载）
   - **Face Detector** (MTCNN): 人脸检测，用于参考图像的预处理裁剪
   - 所有模型权重使用 mmap 模式加载以减少内存峰值

2. **文本驱动视频生成** (`generate_from_text`):
   - ① 参考图像预处理: MTCNN 检测人脸 → 裁剪并resize到512×512 → 生成人脸mask
   - ② 视素→特征: 使用 `VisemeEncoder` 将视素ID序列转为 `(1, T, 19200)` 特征
   - ③ 特征 reshape: `(1, T, 19200)` → `(1, T, 50, 384)` 匹配 EchoMimic pipeline 期望的维度
   - ④ DDIM 去噪: 在潜在空间执行 T 步去噪，每步由 Face Locator 条件指导
   - ⑤ 帧解码: VAE decoder 将去噪后的潜在变量解码为 RGB 像素帧
   - ⑥ 视频编码: 将帧序列编码为 `.mp4` 文件

3. **进度回调**: 支持 `progress_callback(percent, stage, message)` 三层嵌套回调：
   - 扩散步骤进度: 10% → 70%
   - 帧解码进度: 70% → 95%
   - 视频保存: 95% → 100%

4. **音频驱动模式** (`generate`): 保留原始 EchoMimic 的音频驱动接口，用于兼容

**输入**: 参考人脸图像路径 + 视素ID序列 + 目标帧数  
**输出**: `.mp4` 唇动视频文件

---

### 推理流水线集成

**文件**: `lip_sync/inference_pipeline.py`

**函数**: `generate_video_from_text(text, reference_image_path, output_path, config, progress_callback)`

串联整个三阶段流程：
1. 调用 `TextToVisemeProcessor.process_with_duration()` 获取目标帧数，调用 `process()` 获取短视素序列
2. 初始化 `EchoMimicBackend`，调用 `generate_from_text()` 完成视素→特征→视频
3. 支持 `progress_callback` 向前端传递实时进度

**设计关键**: 模型训练时使用短序列（N 个视素），推理时 Cross-Attention Decoder 自适应地将 N 对齐到 T 帧——这就是为什么 `process()` 和 `process_with_duration()` 的输出长度不同但可以配合使用。

---

### 训练管线

**文件**: `lip_sync/train/train_text_to_feature.py`

**训练策略**: 端到端训练 `VisemeEncoder`，使用三种损失的加权组合：

| 损失 | 权重 | 文件 | 用途 |
|------|------|------|------|
| **MSE Loss** | 1.0 | 内置 `nn.MSELoss` | 主损失：最小化预测特征与 GT Whisper 特征的欧氏距离 |
| **LPIPS 感知损失** | 0.1 | `lip_sync/models/aux_renderer.py` → `AuxiliaryRenderer` | 将 19200 维特征解码为 64×64 RGB 帧，与真实帧计算感知相似度 |
| **SyncNet 同步损失** | 0.1 | `lip_sync/models/syncnet.py` → `PretrainedSyncNet` | 基于 Wav2Lip 预训练 SyncNet 的音画同步对比损失 |

**训练优化**:
- AMP 混合精度训练 + Gradient Scaling
- OneCycle LR Scheduler + AdamW 优化器
- 梯度累积（默认4步）支持大有效批次
- torch.compile 加速（自动回退到 eager mode）
- 多 worker DataLoader + pin_memory + prefetch
- Early Stopping（默认 patience=10）

**辅助渲染器** (`AuxiliaryRenderer`):
- 将 `(B, T, 19200)` 特征通过 FC→ConvTranspose2d 上采样到 `(B, T, 3, 64×64)` 低分辨率帧
- 包含 `DummySyncNet` 作为备用的简易同步损失计算

**SyncNet** (`PretrainedSyncNet`):
- 加载 Wav2Lip 的预训练 SyncNet 权重
- 使用 5 帧滑动窗口计算音画对比损失
- 包含 `audio_adapter`（19200→1280）将 Whisper 特征适配到 SyncNet 期望的 mel-spectrogram 形状

---

### 数据预处理

**文件**: `lip_sync/data_utils/preprocess_grid.py`

**处理流程**（多进程并行）:
1. 从 `.align` 对齐文件解析文本
2. 通过 Whisper Audio Processor 提取音频特征 `(T_a, 19200)`
3. 通过 MediaPipe FaceMesh 提取 16 维 FAU 信号
4. 通过 MTCNN 提取 64×64 人脸帧
5. 对齐视频帧数 T_v 与音频帧数 T_a（裁剪或填充）
6. 保存为 `.pt` 文件：`{text, whisper_features, gt_frames, fau_signals}`

**优化**: 
- 跳帧检测（`skip_factor=2`）：每隔N帧才运行 MediaPipe/MTCNN 检测，其余帧复用上次结果
- 多进程并行（`ProcessPoolExecutor`）
- 每个 worker 独立加载模型实例

---

## 项目结构

```
├── lip_sync/                         # 核心代码
│   ├── inference_pipeline.py         # 端到端推理流水线入口
│   ├── models/
│   │   ├── text_to_viseme.py         # Step 1: 文本→视素（G2P + 音素时长预测）
│   │   ├── motion_gen.py             # Step 2: 视素→运动特征（Transformer + Cross-Attention）
│   │   ├── aux_renderer.py           # 辅助解码器（训练用，特征→低分辨率帧）
│   │   ├── syncnet.py                # SyncNet 同步损失计算（训练用）
│   ├── train/
│   │   └── train_text_to_feature.py  # 训练脚本（MSE+LPIPS+SyncNet联合损失）
│   ├── inference/
│   │   └── echomimic_backend.py     # Step 3: EchoMimic 视频生成后端
│   └── data_utils/
│       └── preprocess_grid.py        # GRID 数据集预处理（多进程加速）
├── configs/
│   ├── text_driven_config.yaml       # 训练/推理配置文件
│   ├── inference/                    # EchoMimic 推理配置
│   └── prompts/                      # EchoMimic prompt 配置
├── backend_server.py                 # FastAPI 后端服务（支持 SSE/WebSocket 进度推送）
├── frontend/                         # Vue 3 + Element Plus 前端
│   └── src/components/LipSync.vue
├── train_and_gen.ipynb               # Colab 训练与生成笔记
├── third_party/EchoMimic/            # EchoMimic 第三方依赖（扩散模型管线）
├── pretrained_weights/               # 预训练权重目录（需下载）
└── scripts/download_models.py        # 模型下载脚本
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

参见 `train_and_gen.ipynb`，该笔记包含完整的：
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

---

## 已知问题与改进方向

### 设计层面

1. **[效率] 每次推理重新初始化 EchoMimic 后端**
   - 位置: `lip_sync/inference_pipeline.py` 第 74 行
   - 问题: `generate_video_from_text()` 每次调用都创建新的 `EchoMimicBackend()` 实例，触发完整的模型加载流程（VAE + 两个UNet + FaceLocator + MTCNN），耗时数分钟
   - 建议: 将 `EchoMimicBackend` 实例化移至模块级别，使用单例模式复用

2. **[效率] 预处理器加载完整 EchoMimic 后端仅为了获取 Audio Processor**
   - 位置: `lip_sync/data_utils/preprocess_grid.py` `init_worker()` 函数
   - 问题: 每个多进程 worker 初始化完整的 `EchoMimicBackend()`（加载所有扩散模型），仅为了使用 `audio_processor.audio2feat()` 和 `feature2chunks()`
   - 建议: 直接加载 Whisper 模型（`load_audio_model`），避免加载不相关的 UNet/VAE 权重

3. **[进度逻辑] inference_pipeline 与 backend 内部 progress_callback 重复**
   - 位置: `lip_sync/inference_pipeline.py` 和 `lip_sync/inference/echomimic_backend.py`
   - 问题: inference_pipeline 发送 0% 和 5% 进度后，backend 内部又从 0% 开始发送（preprocess→text_features→diffusion...），导致进度条跳跃
   - 建议: 使用偏移量统一进度范围（如 pipeline 发 0-5%，backend 内部映射为 5-100%）

4. **[默认值不一致] VisemeEncoder 的 d_model 默认值**
   - 位置: `lip_sync/models/motion_gen.py` 第 29 行
   - 问题: `VisemeEncoder.__init__` 默认 `d_model=256`，但在 `echomimic_backend.py` 和训练配置中实际使用 `d_model=512`。虽然运行时会被覆盖，但默认值容易误导
   - 建议: 将默认值改为 512 或与 config 保持一致

### 训练数据层面

5. **[FAU 信号分布] 推理时使用全零 FAU**
   - 位置: `lip_sync/inference/echomimic_backend.py` 第 291 行
   - 问题: 推理时传入 `torch.zeros(...)` 作为 FAU 信号，但训练集中部分样本有真实 FAU、部分为 zeros。如果真实 FAU 样本占比高，全零信号可能与训练分布有偏差
   - 建议: 可考虑训练时提高 zeros 的比例，或推理时使用统计平均的 FAU 向量

6. **[SyncNet 权重缺失处理] 预训练权重不存在时静默失败**
   - 位置: `lip_sync/models/syncnet.py` `PretrainedSyncNet.__init__`
   - 问题: 如果 `pretrained_weights/lipsync_expert.pth` 不存在，`torch.load` 会直接抛出异常，没有友好提示
   - 建议: 添加文件存在性检查和友好的错误消息
