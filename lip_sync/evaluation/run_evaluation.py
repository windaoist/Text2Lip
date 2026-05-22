"""
评测入口：端到端基线对比 + 消融实验。

模式:
  fast  — 使用 AuxiliaryRenderer @ 64x64，不调用扩散模型，快速迭代所有消融变体
  full  — 生成 512x512 全分辨率视频，仅对比 Ours vs Audio-Driven 两个主方法

用法:
    # 快速消融实验（推荐先跑这个）
    python -m lip_sync.evaluation.run_evaluation --mode fast --test-samples 50

    # 完整评测（需要 GPU，FID 在扩散输出上计算）
    python -m lip_sync.evaluation.run_evaluation --mode full --test-samples 10

    # 仅评测（视频已生成，跳过生成阶段）
    python -m lip_sync.evaluation.run_evaluation --mode full --eval-only
"""

import os
import sys
import argparse
import time
import copy
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lip_sync.models.text_to_viseme import TextToVisemeProcessor
from lip_sync.models.motion_gen import VisemeEncoder
from lip_sync.models.aux_renderer import AuxiliaryRenderer
from lip_sync.evaluation.metrics import (
    calculate_fid,
    calculate_psnr,
    calculate_ssim,
    calculate_lmd,
    extract_inception_features,
    extract_lip_landmarks,
    InceptionFeatureExtractor,
    save_metrics_table,
)
from lip_sync.evaluation.ablation import (
    AblationConfig,
    build_ablation_model,
    get_default_ablation_experiments,
)


@dataclass
class EvalConfig:
    # 数据
    data_dir: str = "data/dataset/grid/preprocessed/s1"
    test_split_ratio: float = 0.1
    test_samples: Optional[int] = None
    seed: int = 42
    # 输出
    output_root: str = "output/evaluation"
    results_file: str = "results.csv"
    # 生成
    mode: str = "fast"
    device: str = "cuda"
    fps: int = 25
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg: float = 2.5
    # 权重
    text_model_weights: str = "pretrained_weights/text_driven_model.pth"
    # 评测
    batch_size_fid: int = 32
    compute_lmd: bool = True
    skip_generation: bool = False
    debug: bool = False


class EvaluationRunner:
    """评测运行器。"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"[*] 评测设备: {self.device}")
        if torch.cuda.is_available():
            print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.processor = TextToVisemeProcessor()
        self._test_samples: Optional[List[dict]] = None
        self._text_model: Optional[VisemeEncoder] = None
        self._aux_renderer: Optional[AuxiliaryRenderer] = None
        self._inception_extractor: Optional[InceptionFeatureExtractor] = None
        self.results: Dict[str, Dict] = {}

    # ── 数据 ──

    def _load_test_samples(self) -> List[dict]:
        if self._test_samples is not None:
            return self._test_samples
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        pt_files = sorted(data_dir.glob("*.pt")) or sorted(data_dir.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到 .pt 文件")

        rng = np.random.RandomState(self.config.seed)
        indices = rng.permutation(len(pt_files))
        test_count = (
            min(self.config.test_samples, len(pt_files))
            if self.config.test_samples is not None
            else max(1, int(len(pt_files) * self.config.test_split_ratio))
        )
        self._test_samples = [
            {"path": str(pt_files[i]), "name": pt_files[i].stem, "text": None}
            for i in indices[:test_count]
        ]
        print(f"[*] 数据集共 {len(pt_files)} 个样本，测试集: {len(self._test_samples)} 个")
        return self._test_samples

    def _get_gt_data(self, sample: dict) -> dict:
        data = torch.load(sample["path"], map_location="cpu")
        sample["text"] = data["text"]
        return data

    # ── 模型 ──

    @property
    def text_model(self) -> VisemeEncoder:
        if self._text_model is None:
            print("[*] 加载 VisemeEncoder...")
            model = VisemeEncoder(num_visemes=self.processor.vocab_size, d_model=512, out_dim=19200)
            ckpt = self.config.text_model_weights
            if os.path.exists(ckpt):
                model.load_state_dict(
                    torch.load(ckpt, map_location=self.device, weights_only=False), strict=False
                )
                print(f"[*] 已加载权重: {ckpt}")
            else:
                print(f"[-] 权重文件不存在，使用随机初始化: {ckpt}")
            model.to(self.device).eval()
            self._text_model = model
        return self._text_model

    @property
    def aux_renderer(self) -> AuxiliaryRenderer:
        if self._aux_renderer is None:
            self._aux_renderer = AuxiliaryRenderer().to(self.device).eval()
        return self._aux_renderer

    @property
    def inception_extractor(self) -> InceptionFeatureExtractor:
        if self._inception_extractor is None:
            self._inception_extractor = InceptionFeatureExtractor(device=self.device)
        return self._inception_extractor

    # ── fast 模式 ──

    def _evaluate_fast_mode(self) -> Dict[str, Dict]:
        """
        快速模式：VisemeEncoder + AuxiliaryRenderer @ 64x64。
        不调用扩散模型，适合快速迭代所有消融变体。
        """
        print("\n" + "=" * 60)
        print("快速评测模式 (AuxiliaryRenderer @ 64x64)")
        print("=" * 60)

        samples = self._load_test_samples()
        results = {}

        # ── 加载全部 GT ──
        print("[*] 加载测试集真实帧...")
        all_gt, all_feat = [], []
        for s in samples:
            d = self._get_gt_data(s)
            all_gt.append(d["gt_frames"])
            all_feat.append(d["whisper_features"])
        gt_frames_all = torch.cat(all_gt, dim=0)  # (total_T, 3, 64, 64)

        print("[*] 计算真实帧 Inception 特征...")
        real_feats = extract_inception_features(
            gt_frames_all, self.inception_extractor, self.device, self.config.batch_size_fid
        )

        # ── 辅助渲染器基线：Whisper 特征直接解码 ──
        results.update(self._run_fast_variant(
            "AuxRenderer (Whisper→64x64)",
            samples, gt_frames_all, real_feats,
            lambda s, gt: gt["whisper_features"].unsqueeze(0),
        ))

        # ── 完整文本驱动模型 ──
        def full_model_fn(s, gt):
            text, T = gt["text"], gt["whisper_features"].shape[0]
            ids = self.processor.process(text).unsqueeze(0).to(self.device)
            fau = gt.get("fau_signals")
            if fau is not None:
                fau = fau.unsqueeze(0).to(self.device)
            return self.text_model(ids, target_frames_len=T, fau_signals=fau)

        results.update(self._run_fast_variant(
            "Ours (Full)", samples, gt_frames_all, real_feats, full_model_fn,
        ))

        # ── 消融变体 ──
        for name, ablation_cfg in get_default_ablation_experiments():
            if name == "Full (Baseline)":
                continue

            model_abl = build_ablation_model(ablation_cfg, self.text_model)

            def make_abl_fn(cfg, model):
                def fn(s, gt):
                    text, T = gt["text"], gt["whisper_features"].shape[0]
                    ids = self.processor.process(text).unsqueeze(0).to(self.device)
                    target_T = ids.shape[1] * 3 if not cfg.use_duration else T
                    fau = gt.get("fau_signals")
                    if fau is not None and cfg.use_fau:
                        fau = fau.unsqueeze(0).to(self.device)
                    else:
                        fau = None
                    return model(ids, target_frames_len=target_T, fau_signals=fau)
                return fn

            results.update(self._run_fast_variant(
                name, samples, gt_frames_all, real_feats, make_abl_fn(ablation_cfg, model_abl),
            ))

        return results

    def _run_fast_variant(
        self,
        name: str,
        samples: List[dict],
        gt_frames_all: torch.Tensor,
        real_feats: np.ndarray,
        feature_fn,
    ) -> Dict[str, Dict]:
        """运行一个 fast 模式的变体，返回 {name: metrics}。"""
        print(f"\n[*] 变体: {name}")
        all_gen, p_list, s_list, lm_gen, lm_gt = [], [], [], [], []

        with torch.no_grad():
            for sidx, s in enumerate(samples):
                gt = self._get_gt_data(s)
                feats = feature_fn(s, gt)                         # (1, T, 19200)
                pred = self.aux_renderer(feats)[0]                 # (T, 3, 64, 64)

                T = min(pred.shape[0], gt["gt_frames"].shape[0])
                gcrop, gt_crop = pred[:T], gt["gt_frames"][:T]

                # 逐帧指标（每样本一个均值）
                p = calculate_psnr(gcrop.unsqueeze(0), gt_crop.unsqueeze(0)).mean().item()
                s = calculate_ssim(gcrop.unsqueeze(0), gt_crop.unsqueeze(0)).mean().item()
                p_list.append(p)
                s_list.append(s)
                all_gen.append(gcrop)

                if self.config.compute_lmd and sidx % 5 == 0:
                    lm_gen.extend(extract_lip_landmarks(gcrop))
                    lm_gt.extend(extract_lip_landmarks(gt_crop))

        gen_all = torch.cat(all_gen, dim=0)
        print(f"  计算 FID ({gen_all.shape[0]} 帧)...")
        gen_feats = extract_inception_features(
            gen_all, self.inception_extractor, self.device, self.config.batch_size_fid
        )
        fid = calculate_fid(real_feats, gen_feats)

        metrics = {"FID": fid, "PSNR": float(np.mean(p_list)), "SSIM": float(np.mean(s_list))}
        if lm_gen:
            lmd, lmd_std, valid = calculate_lmd(lm_gen, lm_gt)
            metrics.update({"LMD": lmd, "LMD_std": lmd_std, "ValidRatio": valid})

        print(f"  FID={fid:.4f}  PSNR={np.mean(p_list):.2f}  SSIM={np.mean(s_list):.4f}", end="")
        if "LMD" in metrics:
            print(f"  LMD={metrics['LMD']:.4f}", end="")
        print()

        return {name: metrics}

    # ── full 模式 ──

    def _evaluate_full_mode(self) -> Dict[str, Dict]:
        """
        完整模式：生成 512x512 扩散视频，计算 FID。
        注意：扩散生成非常慢，仅对比 Ours vs Audio-Driven。
        """
        print("\n" + "=" * 60)
        print("完整评测模式 (512x512 扩散模型)")
        print("=" * 60)

        samples = self._load_test_samples()
        results = {}

        methods = [
            ("ours_full", None),
            ("audio_driven_echomimic", None),
        ]

        # 生成
        if not self.config.skip_generation:
            for method_key, _ in methods:
                print(f"\n[*] 正在生成 [{method_key}]...")
                for sidx, s in enumerate(samples):
                    if sidx % 5 == 0:
                        print(f"  [{sidx}/{len(samples)}] {s['name']}")
                    self._gen_video(s, method_key)

        # GT 特征
        print("\n[*] 加载真实帧...")
        all_gt = [self._get_gt_data(s)["gt_frames"] for s in samples]
        gt_all = torch.cat(all_gt, dim=0)
        real_feats = extract_inception_features(
            gt_all, self.inception_extractor, self.device, self.config.batch_size_fid
        )

        # 评测
        for method_key, _ in methods:
            print(f"\n[*] 评测 [{method_key}]...")
            method_dir = self.output_root / method_key
            if not method_dir.exists():
                print(f"  [-] 目录不存在，跳过")
                continue

            frames_list = []
            for s in samples:
                vp = method_dir / f"{s['name']}.mp4"
                if vp.exists() and vp.stat().st_size > 1024:
                    try:
                        frames_list.append(self._load_video(str(vp)))
                    except Exception as e:
                        print(f"  [-] 加载失败 {vp.name}: {e}")

            if not frames_list:
                continue

            gen_all = torch.cat(frames_list, dim=0)
            print(f"  计算 FID ({gen_all.shape[0]} 帧)...")
            gen_feats = extract_inception_features(
                gen_all, self.inception_extractor, self.device, self.config.batch_size_fid
            )
            fid = calculate_fid(real_feats, gen_feats)
            results[method_key] = {"FID": fid}
            print(f"  FID = {fid:.4f}")

        return results

    def _gen_video(self, sample: dict, method_key: str) -> Optional[str]:
        """为指定方法生成单个样本的视频。"""
        out_dir = self.output_root / method_key
        out_dir.mkdir(parents=True, exist_ok=True)
        vpath = str(out_dir / f"{sample['name']}.mp4")

        if os.path.exists(vpath) and os.path.getsize(vpath) > 1024:
            return vpath

        from lip_sync.inference_pipeline import generate_video_from_text

        gt = self._get_gt_data(sample)

        try:
            if method_key == "audio_driven_echomimic":
                audio_path = self._find_audio(sample["name"])
                if not audio_path or not os.path.exists(audio_path):
                    print(f"  [-] 未找到音频: {sample['name']}")
                    return None
                from lip_sync.inference.echomimic_backend import EchoMimicBackend
                backend = EchoMimicBackend()
                return backend.generate(
                    self._get_ref_image(), audio_path, vpath,
                    width=self.config.width, height=self.config.height,
                    length=gt["whisper_features"].shape[0],
                    steps=self.config.steps, cfg=self.config.cfg, fps=self.config.fps,
                )
            else:
                return generate_video_from_text(gt["text"], self._get_ref_image(), vpath)

        except Exception as e:
            print(f"  [-] 生成失败 [{sample['name']}]: {e}")
            if self.config.debug:
                import traceback
                traceback.print_exc()
            return None

    def _load_video(self, path: str) -> torch.Tensor:
        """加载视频文件为 (T, 3, H, W) 张量，范围 [-1, 1]。"""
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 127.5 - 1.0)
        cap.release()
        return torch.stack(frames)

    def _get_ref_image(self) -> str:
        ref_dir = Path("data/reference")
        for c in ref_dir.glob("*"):
            if c.suffix.lower() in (".png", ".jpg", ".jpeg"):
                return str(c)
        return str(ref_dir / "test_face.png")

    def _find_audio(self, sample_name: str) -> Optional[str]:
        for d in [Path("data/dataset/grid/s1"), Path("data/dataset/grid/s1_audio")]:
            if d.exists():
                p = d / f"{sample_name}.wav"
                if p.exists():
                    return str(p)
        return None

    # ── 主入口 ──

    def run_all(self) -> Dict[str, Dict]:
        t0 = time.time()
        if self.config.mode == "fast":
            self.results = self._evaluate_fast_mode()
        elif self.config.mode == "full":
            self.results = self._evaluate_full_mode()
        else:
            raise ValueError(f"未知模式: {self.config.mode}")

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"评测完成，耗时 {elapsed:.1f}s")
        print("=" * 60)

        if self.results:
            save_metrics_table(self.results, str(self.output_root / self.config.results_file))
            save_metrics_table(self.results, str(self.output_root / "results.md"), fmt="markdown")

        return self.results


def main():
    p = argparse.ArgumentParser(description="Text2Lip 评测脚本")
    p.add_argument("--mode", choices=["fast", "full"], default="fast",
                   help="fast=64x64辅助渲染器 | full=512x512全扩散")
    p.add_argument("--test-samples", type=int, default=None)
    p.add_argument("--test-split-ratio", type=float, default=0.1)
    p.add_argument("--eval-only", action="store_true", help="跳过生成，仅评测已有视频")
    p.add_argument("--data-dir", default="data/dataset/grid/preprocessed/s1")
    p.add_argument("--weights", default="pretrained_weights/text_driven_model.pth")
    p.add_argument("--output", default="output/evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-lmd", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    cfg = EvalConfig(
        mode=args.mode, test_samples=args.test_samples,
        test_split_ratio=args.test_split_ratio, data_dir=args.data_dir,
        text_model_weights=args.weights, output_root=args.output,
        batch_size_fid=args.batch_size, compute_lmd=not args.no_lmd,
        skip_generation=args.eval_only, device=args.device,
        seed=args.seed, debug=args.debug,
    )
    EvaluationRunner(cfg).run_all()


if __name__ == "__main__":
    main()
