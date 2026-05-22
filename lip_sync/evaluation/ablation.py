"""
消融实验模块：定义可组合的消融配置，通过禁用特定模块来测试其对最终效果的影响。

每个消融变体对应一个 AblationConfig，可以组合使用（如同时禁用 Duration + FAU）。

基本用法:
    config = AblationConfig(use_duration=False, use_cross_attention=False)
    model = build_ablation_model(config, base_model)
    output = model(viseme_ids, target_frames_len=T)
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from lip_sync.models.motion_gen import VisemeEncoder, PositionalEncoding


class AblationType(Enum):
    """消融类型枚举"""
    FULL = "full"                     # 完整模型
    NO_DURATION = "no_duration"       # 无音素时长预测
    NO_CROSS_ATTN = "no_cross_attn"   # 无交叉注意力
    NO_FAU = "no_fau"                 # 无 FAU 微表情条件
    NO_POS_ENCODING = "no_pos_enc"    # 无位置编码
    REDUCED_LAYERS = "reduced_layers"  # 减少 Transformer 层数
    NO_TRANSFORMER = "no_transformer"  # 无自注意力编码器，直接映射


ABLATION_DESCRIPTIONS = {
    AblationType.FULL:              "完整模型（基线）",
    AblationType.NO_DURATION:       "消融：固定帧率（3帧/音素，无语言学时长预测）",
    AblationType.NO_CROSS_ATTN:     "消融：重复展开替代交叉注意力",
    AblationType.NO_FAU:            "消融：去除 FAU 微表情条件注入",
    AblationType.NO_POS_ENCODING:   "消融：去除位置编码",
    AblationType.REDUCED_LAYERS:    "消融：减少编码器层数 (4→1) 和解码器层数 (2→1)",
    AblationType.NO_TRANSFORMER:    "消融：去除自注意力编码器，仅用嵌入+线性投影",
}


@dataclass
class AblationConfig:
    """
    消融配置。

    可以通过构造参数精确控制每个模块的开关，也可以通过 from_type 工厂方法快速创建。
    """
    use_duration: bool = True           # 是否使用语言学时长预测
    use_cross_attention: bool = True    # 是否使用交叉注意力解码器
    use_fau: bool = True                # 是否使用 FAU 微表情条件
    use_positional_encoding: bool = True  # 是否使用位置编码
    num_encoder_layers: int = 4         # Transformer 编码器层数
    num_decoder_layers: int = 2         # Transformer 解码器层数
    use_self_attention: bool = True     # 是否使用自注意力编码器

    @classmethod
    def from_type(cls, at: AblationType) -> "AblationConfig":
        """从消融类型创建预定义配置。"""
        mapping = {
            AblationType.FULL:              cls(),
            AblationType.NO_DURATION:       cls(use_duration=False),
            AblationType.NO_CROSS_ATTN:     cls(use_cross_attention=False),
            AblationType.NO_FAU:            cls(use_fau=False),
            AblationType.NO_POS_ENCODING:   cls(use_positional_encoding=False),
            AblationType.REDUCED_LAYERS:    cls(num_encoder_layers=1, num_decoder_layers=1),
            AblationType.NO_TRANSFORMER:    cls(use_self_attention=False, use_cross_attention=False),
        }
        return mapping[at]

    def description(self) -> str:
        """返回人类可读的消融描述。"""
        parts = []
        if not self.use_duration:
            parts.append("固定帧率")
        if not self.use_cross_attention:
            parts.append("无交叉注意力")
        if not self.use_fau:
            parts.append("无FAU")
        if not self.use_positional_encoding:
            parts.append("无位置编码")
        if self.num_encoder_layers != 4 or self.num_decoder_layers != 2:
            parts.append(
                f"减少层数(enc={self.num_encoder_layers},dec={self.num_decoder_layers})")
        if not self.use_self_attention:
            parts.append("无自注意力")
        return "完整模型" if not parts else "消融: " + ", ".join(parts)


# ============================================================================
# 无交叉注意力的替代：直接重复展开
# ============================================================================

class DirectExpansionDecoder(nn.Module):
    """
    用简单的线性投影 + 重复展开替代交叉注意力。
    将 N 个视素嵌入直接线性投影到 d_model，再重复到 T 帧。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, memory: torch.Tensor, target_frames_len: int) -> torch.Tensor:
        """
        memory: (B, N, d_model) 编码器输出
        返回:   (B, T, d_model)
        """
        B, N, D = memory.shape
        # 对每个视素嵌入做线性变换
        decoded = self.proj(memory)  # (B, N, D)
        # 均匀重复展开: 每个视素占据 ceil(T/N) 帧
        repeats = math.ceil(target_frames_len / N)
        decoded = decoded.repeat(1, repeats, 1)  # (B, N*repeats, D)
        return decoded[:, :target_frames_len, :]


# ============================================================================
# 构建消融模型
# ============================================================================

def build_ablation_model(
    cfg: AblationConfig,
    base_model: VisemeEncoder,
    processor=None,
) -> VisemeEncoder:
    """
    根据消融配置修改 VisemeEncoder。

    策略：深拷贝 base_model 后再修改，避免影响原始模型。
    对需要替换的结构，直接替换对应子模块。
    """
    # 深拷贝，避免影响原始模型（多次消融实验共享同一 base_model 时需要）
    base_model = copy.deepcopy(base_model)
    device = next(base_model.parameters()).device

    # ── 消融：无自注意力编码器 ──
    if not cfg.use_self_attention:
        base_model.transformer_encoder = nn.Identity()

    # ── 消融：无位置编码 ──
    if not cfg.use_positional_encoding:
        base_model.pos_encoder = nn.Identity()

    # ── 消融：减少编码器层数 ──
    if cfg.num_encoder_layers != 4 and cfg.use_self_attention:
        from torch.nn import TransformerEncoderLayer
        d_model = base_model.d_model
        nhead = base_model.transformer_encoder.layers[0].self_attn.num_heads
        new_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4,
        )
        base_model.transformer_encoder = nn.TransformerEncoder(
            new_encoder_layer, num_layers=cfg.num_encoder_layers,
        ).to(device)

    # ── 消融：无交叉注意力（替换为 DirectExpansionDecoder） ──
    if not cfg.use_cross_attention:
        base_model.transformer_decoder = DirectExpansionDecoder(
            d_model=base_model.d_model).to(device)

    # ── 消融：减少解码器层数（仅在仍使用交叉注意力时生效） ──
    if cfg.use_cross_attention and cfg.num_decoder_layers != 2:
        from torch.nn import TransformerDecoderLayer
        d_model = base_model.d_model
        nhead = base_model.transformer_decoder.layers[0].multihead_attn.num_heads
        new_decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4,
        )
        base_model.transformer_decoder = nn.TransformerDecoder(
            new_decoder_layer, num_layers=cfg.num_decoder_layers,
        ).to(device)

    # ── 消融：无 FAU（在 forward 中拦截） ──
    # FAU 条件是通过检查 fau_signals 是否为 None 来控制的。
    # 这里我们通过 monkey-patch forward 来强制 fau_signals=None
    if not cfg.use_fau:
        original_forward = base_model.forward

        def patched_forward(viseme_ids, target_frames_len=None,
                            viseme_padding_mask=None, fau_signals=None,
                            **kwargs):
            # 强制 fau_signals 为 None
            return original_forward(
                viseme_ids,
                target_frames_len=target_frames_len,
                viseme_padding_mask=viseme_padding_mask,
                fau_signals=None,
                **kwargs
            )
        base_model.forward = patched_forward

    return base_model


# ============================================================================
# 消融推理流水线
# ============================================================================

def create_ablation_inference_fn(
    ablation_cfg: AblationConfig,
    text_model: VisemeEncoder,
    processor=None,
) -> Callable:
    """
    创建消融变体的推理函数。

    返回一个可调用对象，签名与 VisemeEncoder.forward 兼容，
    但会按照 ablation_cfg 修改行为。

    Args:
        ablation_cfg: 消融配置
        text_model:   训练好的 VisemeEncoder
        processor:    可选的 TextToVisemeProcessor（no_duration 时需要）

    Returns:
        callable(viseme_ids, target_frames_len, ...) → features (B, T, 19200)
    """
    model = build_ablation_model(ablation_cfg, text_model, processor)

    def infer(viseme_ids, target_frames_len=None, viseme_padding_mask=None,
              fau_signals=None, **kwargs):
        # no_duration: target_frames_len 由固定的每视素帧数决定
        if not ablation_cfg.use_duration and target_frames_len is None:
            # 默认每视素 3 帧（固定帧率）
            target_frames_len = viseme_ids.shape[1] * 3
        return model(
            viseme_ids,
            target_frames_len=target_frames_len,
            viseme_padding_mask=viseme_padding_mask,
            fau_signals=fau_signals if ablation_cfg.use_fau else None,
        )

    return infer


# ============================================================================
# 工具：预定义消融实验列表
# ============================================================================

def get_default_ablation_experiments() -> List[tuple[str, AblationConfig]]:
    """返回默认的消融实验列表（名称 → 配置）。"""
    return [
        ("Full (Baseline)", AblationConfig()),
        ("w/o Duration Prediction", AblationConfig(use_duration=False)),
        ("w/o Cross-Attention", AblationConfig(use_cross_attention=False)),
        ("w/o FAU Conditioning", AblationConfig(use_fau=False)),
        ("w/o Positional Encoding", AblationConfig(use_positional_encoding=False)),
        ("Reduced Layers (enc=1, dec=1)", AblationConfig(
            num_encoder_layers=1, num_decoder_layers=1)),
        ("No Transformer (Embed→Proj→Repeat)", AblationConfig(
            use_self_attention=False, use_cross_attention=False,
            use_positional_encoding=False)),
    ]
