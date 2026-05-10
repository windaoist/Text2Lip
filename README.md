# Text-Driven Video Generation Prototype

这是一个纯文本驱动视频生成项目的原型，已整理并实现为 1-2-3 三步骤流水线。

## 项目结构

```
E:\playground\
├── configs\            # 配置文件 (如 environment.yml)
├── data\               # 数据文件
│   ├── reference\      # 参考人脸图像
│   ├── audio\          # 生成的音频文件
│   ├── asserts\        # 原始资源 (AVDigits 等)
│   └── dataset\        # 训练/测试数据集
├── output\             # 输出结果
├── src\                # 源代码
│   ├── step1_text_to_viseme.py   # 步骤 1: 文本 -> 视位 (Viseme)
│   ├── step2_motion_gen.py       # 步骤 2: 跨模态面部运动生成
│   ├── step3_renderer.py         # 步骤 3: 视频渲染
│   ├── unified_pipeline.py       # 统一流水线入口 (1-2-3 整合)
│   ├── text_driven_prototype.py  # 基于 EchoMimic 的端到端原型
│   └── locations.py / test.py    # 原始脚本备份/组件库
└── third_party\        # 第三方库 (如 EchoMimic)
```

## 核心步骤说明

1.  **文本-语音转换 (TTS)**: 使用 `edge-tts` 将输入文本转换为高质量语音文件。
2.  **音频-视频生成 (EchoMimic Backend)**: 使用 EchoMimic 作为核心后端，提取语音特征并驱动参考人脸，生成具有真实口型和表情的视频。
3.  **原始 3 步骤原型 (Fallback)**: 作为研究参考，保留了 文本->视素->运动->渲染 的轻量化实验路径。

## 如何运行

### 运行集成了 EchoMimic 的真实流水线
```bash
python src/text_driven_prototype.py
```
该脚本将：
- 调用 `edge-tts` 生成语音。
- 自动加载 `third_party/EchoMimic` 中的预训练模型。
- 生成带音频的 `.mp4` 视频文件。

### 运行原始 3 步骤架构原型 (Dummy)
```bash
python src/unified_pipeline.py
```

## 注意事项
- 当前 Step 2 和 Step 3 采用的是架构原型实现，模型权重需进一步训练或加载预训练模型。
- 如果需要真实的视频生成效果，请确保 `third_party/EchoMimic` 环境配置正确，并运行 `src/text_driven_prototype.py`。
