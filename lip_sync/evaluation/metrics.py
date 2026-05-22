"""
评测指标模块：提供 FID / PSNR / SSIM / LMD / SyncNet 等指标的计算。

所有指标均支持 (B, T, C, H, W) 批量化输入，
退化到单帧或单视频时自动处理维度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
import warnings

# ============================================================================
# FID — Fréchet Inception Distance
# ============================================================================

class InceptionFeatureExtractor(nn.Module):
    """使用预训练 Inception-v3 提取 2048 维特征用于 FID 计算。"""

    def __init__(self, device="cuda"):
        super().__init__()
        try:
            import torchvision.models as models
            self.inception = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1
            ).eval().to(device)
        except Exception:
            try:
                import torchvision.models as models
                self.inception = models.inception_v3(pretrained=True).eval().to(device)
            except Exception as e:
                raise ImportError(
                    f"无法加载 Inception-v3: {e}。请确保 torchvision 已安装。"
                )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W), RGB, float32, 范围 [0, 1]
        返回: (B, 2048) 特征向量
        """
        # 确保输入与模型在同一设备
        if x.device != next(self.inception.parameters()).device:
            x = x.to(next(self.inception.parameters()).device)

        # Inception 要求输入至少 299x299
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # 归一化到 Inception 期望的分布
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 手动走 Inception-v3 前向，在 fc 层之前截断
        # Inception3.forward 结构: Conv2d 特征层 → AdaptiveAvgPool2d → Dropout → flatten → fc
        # 我们确保走完 Mixed_7c 后自己 pool，不进入 fc
        x = self.inception._transform_input(x)
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        # 池化 + 展平，不进入 fc
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x  # (B, 2048)


def _compute_fid_from_stats(mu1: np.ndarray, sigma1: np.ndarray,
                            mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """给定两组统计量计算 FID。"""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # 处理数值误差
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def calculate_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """
    计算 FID。

    Args:
        real_features: (N, D) 真实图像特征
        gen_features:  (M, D) 生成图像特征

    Returns:
        FID 标量值（越低越好）
    """
    from scipy.linalg import sqrtm

    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    return _compute_fid_from_stats(mu1, sigma1, mu2, sigma2)


@torch.no_grad()
def extract_inception_features(
    frames: torch.Tensor,
    extractor: Optional[InceptionFeatureExtractor] = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """
    从帧序列批量提取 Inception 特征。

    Args:
        frames: (B, T, C, H, W) 或 (N, C, H, W)，范围 [-1, 1] 或 [0, 1]
        extractor: 可复用的特征提取器
        device: 计算设备
        batch_size: 批大小

    Returns:
        (总帧数, 2048) numpy 特征数组
    """
    if extractor is None:
        extractor = InceptionFeatureExtractor(device=device)

    # 处理维度: (B, T, C, H, W) → (B*T, C, H, W)
    if frames.dim() == 5:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)
    else:
        frames_flat = frames

    # 归一化到 [0, 1]（如果当前是 [-1, 1]）
    if frames_flat.min() < -0.5:
        frames_flat = (frames_flat + 1.0) / 2.0
    frames_flat = frames_flat.clamp(0.0, 1.0)

    all_features = []
    for i in range(0, len(frames_flat), batch_size):
        batch = frames_flat[i:i + batch_size].to(device)
        feats = extractor(batch)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ============================================================================
# PSNR — Peak Signal-to-Noise Ratio
# ============================================================================

def calculate_psnr(
    generated: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 2.0,
) -> torch.Tensor:
    """
    计算 PSNR。

    Args:
        generated: (B, T, C, H, W) 生成帧，范围 [-1, 1]
        target:    (B, T, C, H, W) 真实帧，范围 [-1, 1]
        max_val:   像素最大值（默认 2.0 对应 [-1, 1] 范围）

    Returns:
        (B, T) 每帧的 PSNR 值
    """
    mse = F.mse_loss(generated, target, reduction="none")  # (B, T, C, H, W)
    mse = mse.mean(dim=[2, 3, 4])  # (B, T)
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr  # (B, T)


# ============================================================================
# SSIM — Structural Similarity Index
# ============================================================================

def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """生成 1D 高斯窗口。"""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def calculate_ssim(
    generated: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """
    计算 SSIM。

    Args:
        generated: (B, T, C, H, W)
        target:    (B, T, C, H, W)

    Returns:
        (B, T) 每帧 SSIM
    """
    device = generated.device
    B, T, C, H, W = generated.shape

    # 构建高斯权重
    window_1d = _gaussian_window(window_size, sigma, device)  # (W,)
    window_2d = window_1d[:, None] * window_1d[None, :]       # (W, W)
    window = window_2d.expand(C, 1, window_size, window_size)  # (C, 1, W, W)

    padding = window_size // 2

    # 展平 B,T 维度
    gen = generated.view(B * T, C, H, W)
    gt = target.view(B * T, C, H, W)

    def _conv2d(x):
        return F.conv2d(x, window, groups=C, padding=padding)

    mu1 = _conv2d(gen)
    mu2 = _conv2d(gt)
    sigma1_sq = _conv2d(gen ** 2) - mu1 ** 2
    sigma2_sq = _conv2d(gt ** 2) - mu2 ** 2
    sigma12 = _conv2d(gen * gt) - mu1 * mu2

    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_per_frame = ssim_map.view(B, T, C, -1).mean(dim=[2, 3])
    return ssim_per_frame  # (B, T)


# ============================================================================
# LMD — Landmark Distance (口型关键点距离)
# ============================================================================

def extract_lip_landmarks(
    frames: torch.Tensor,
    landmark_model=None,
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    用 MediaPipe FaceMesh 提取每帧的口唇关键点。

    Args:
        frames: (T, C, H, W) RGB 帧，范围 [0, 1] 或 [-1, 1]
        landmark_model: 可选的 MediaPipe FaceMesh 实例
        device: 计算设备（仅用于输入准备）

    Returns:
        List[T 个 numpy 数组]，每个形状 (20, 2) 表示 20 个口唇关键点 (x, y)
        若某帧未检测到人脸，对应数组为 None
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("需要安装 mediapipe: pip install mediapipe")

    if landmark_model is None:
        mp_face_mesh = mp.solutions.face_mesh
        landmark_model = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    # GRID 数据集的标准口唇关键点索引（MediaPipe FaceMesh）
    # 上唇: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409
    # 下唇: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
    LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                   146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    # 转成 numpy 并归一化到 [0, 255] uint8
    if frames.dim() == 4:
        frames_np = frames.detach().cpu()
        if frames_np.min() < -0.5:
            frames_np = (frames_np + 1.0) / 2.0
        frames_np = (frames_np.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)
    else:
        frames_np = frames

    results = []
    for i in range(len(frames_np)):
        rgb = frames_np[i]
        mp_result = landmark_model.process(rgb)
        if mp_result and mp_result.multi_face_landmarks:
            landmarks = mp_result.multi_face_landmarks[0].landmark
            lip_pts = np.array([[landmarks[idx].x, landmarks[idx].y]
                                for idx in LIP_INDICES], dtype=np.float32)
            results.append(lip_pts)
        else:
            results.append(None)

    return results


def calculate_lmd(
    gen_landmarks: List[Union[np.ndarray, None]],
    gt_landmarks: List[Union[np.ndarray, None]],
) -> Tuple[float, float, float]:
    """
    计算 Lip Landmark Distance (LMD) 及其变体。

    Args:
        gen_landmarks: 生成帧的口唇关键点列表
        gt_landmarks:  真实帧的口唇关键点列表

    Returns:
        (LMD, LMD_std, valid_ratio) 三元组:
          - LMD: 平均欧氏距离（仅有效帧对）
          - LMD_std: 标准差
          - valid_ratio: 成功检测到人脸的比例
    """
    distances = []
    valid = 0
    total = min(len(gen_landmarks), len(gt_landmarks))

    for i in range(total):
        g = gen_landmarks[i]
        t = gt_landmarks[i]
        if g is not None and t is not None:
            dist = np.sqrt(np.sum((g - t) ** 2, axis=1)).mean()
            distances.append(dist)
            valid += 1

    if valid == 0:
        return float("inf"), float("inf"), 0.0

    return float(np.mean(distances)), float(np.std(distances)), valid / total


# ============================================================================
# SyncNet Confidence Score (音画同步置信度)
# ============================================================================

def calculate_syncnet_score(
    video_frames: torch.Tensor,
    audio_features: torch.Tensor,
    syncnet_model=None,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    使用预训练 SyncNet 计算音画同步分数。

    Args:
        video_frames:   (T, C, H, W) 视频帧
        audio_features: (T, 19200) Whisper 特征
        syncnet_model: 可选 PretrainedSyncNet 实例
        device: 计算设备

    Returns:
        (avg_confidence, avg_offset) — 平均置信度（越高越好）和平均偏移
    """
    from lip_sync.models.syncnet import PretrainedSyncNet

    if syncnet_model is None:
        syncnet_model = PretrainedSyncNet().to(device)
        syncnet_model.eval()

    T = video_frames.shape[0]
    window_size = 5
    if T < window_size:
        return 0.0, 0.0

    video_frames = video_frames.unsqueeze(0)  # (1, T, C, H, W)
    audio_features = audio_features.unsqueeze(0)  # (1, T, 19200)

    scores = []
    with torch.no_grad():
        for i in range(T - window_size + 1):
            v_win = video_frames[:, i:i + window_size, :, :, :]  # (1, 5, C, H, W)
            a_center = audio_features[:, i + window_size // 2, :]  # (1, 19200)

            B = 1
            _, C, H, W = v_win.shape[1:]
            # 裁剪下半脸并 resize 到 96x96
            half_h = H // 2
            v_cropped = v_win[:, :, :, half_h:, :]  # (1, 5, C, H/2, W)
            v_flat = v_cropped.reshape(B * window_size, C, H // 2, W)
            v_resized = F.interpolate(v_flat, size=(96, 96), mode="bilinear", align_corners=False)
            v_input = v_resized.reshape(B, window_size, C, 96, 96).permute(0, 2, 1, 3, 4)
            v_input = v_input.reshape(B, C * window_size, 96, 96)  # (1, 15, 96, 96)

            # 音频特征投影
            a_proj = syncnet_model.audio_adapter(a_center)  # (1, 80*16)
            a_input = a_proj.view(1, 1, 80, 16)

            a_emb, v_emb = syncnet_model.syncnet(a_input, v_input)
            v_emb = syncnet_model.face_adapter(v_emb)

            sim = F.cosine_similarity(v_emb, a_emb)
            scores.append(sim.item())

    if len(scores) == 0:
        return 0.0, 0.0

    return float(np.mean(scores)), float(np.std(scores))


# ============================================================================
# 帧预处理工具
# ============================================================================

def preprocess_frames_for_fid(
    frames: torch.Tensor,
    size: int = 299,
) -> torch.Tensor:
    """
    将帧预处理为 FID 所需格式：
    - 归一化到 [0, 1]
    - 调整大小到 299x299
    - 确保 RGB 3 通道

    Args:
        frames: (B, T, C, H, W) 或 (N, C, H, W)
        size: 目标大小

    Returns:
        (N', 3, size, size) float32 张量，范围 [0, 1]
    """
    if frames.dim() == 5:
        B, T, C, H, W = frames.shape
        frames = frames.reshape(B * T, C, H, W)

    # 灰度转 RGB
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)

    # [-1, 1] → [0, 1]
    if frames.min() < -0.5:
        frames = (frames + 1.0) / 2.0
    frames = frames.clamp(0.0, 1.0)

    # resize
    if frames.shape[-1] != size:
        frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)

    return frames.float()


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> torch.Tensor:
    """
    从视频文件加载帧为张量。

    Returns:
        (T, C, H, W) float32 张量，范围 [-1, 1]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3)
        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1)  # (3, H, W)
        frame_tensor = frame_tensor / 127.5 - 1.0  # [0, 255] → [-1, 1]
        frames.append(frame_tensor)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return torch.stack(frames)


def save_metrics_table(
    results: dict,
    output_path: str,
    fmt: str = "markdown",
):
    """
    将评测结果保存为表格。

    Args:
        results: {
            "method_name": {
                "FID": float,
                "PSNR": float,
                "SSIM": float,
                "LMD": float,
                "SyncNet": float,
            },
            ...
        }
        output_path: 输出文件路径
        fmt: "markdown" 或 "csv"
    """
    import pandas as pd

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Method"

    # 重新排列列顺序
    preferred_order = ["FID", "PSNR", "SSIM", "LMD", "LMD_std", "ValidRatio", "SyncNet"]
    available_cols = [c for c in preferred_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in preferred_order]
    df = df[available_cols + extra_cols]

    if fmt == "markdown":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Results\n\n")
            f.write(df.to_markdown(floatfmt=".4f"))
            f.write("\n")
    elif fmt == "csv":
        df.to_csv(output_path, float_format="%.4f")
    else:
        df.to_csv(output_path, float_format="%.4f")

    print(f"[*] 评测结果已保存至: {output_path}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
