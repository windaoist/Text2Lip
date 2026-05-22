"""
评测入口：端到端基线对比 + 消融实验。

模式:
  fast  — 使用 AuxiliaryRenderer @ 64x64，不调用扩散模型，快速迭代所有消融变体
  full  — 生成 512x512 全分辨率视频，计算 FID / PSNR / SSIM / LMD / SyncNet

用法:
    # 快速消融实验（推荐先跑这个）
    python -m lip_sync.evaluation.run_evaluation --mode fast --test-samples 50

    # 完整评测（需要 GPU，所有指标全开）
    python -m lip_sync.evaluation.run_evaluation --mode full --test-samples 10

    # 仅评测（视频已生成，跳过生成阶段）
    python -m lip_sync.evaluation.run_evaluation --mode full --eval-only

    # 跳过 LMD 或 SyncNet（若未安装对应依赖）
    python -m lip_sync.evaluation.run_evaluation --mode full --eval-only --no-lmd --no-syncnet
    python -m lip_sync.evaluation.run_evaluation --mode full --eval-only --lmd-sample-every 5
"""

from lip_sync.evaluation.ablation import (
    AblationConfig,
    build_ablation_model,
    get_default_ablation_experiments,
)
from lip_sync.evaluation.metrics import (
    calculate_fid,
    calculate_psnr,
    calculate_ssim,
    calculate_lmd,
    calculate_syncnet_score,
    extract_inception_features,
    extract_lip_landmarks,
    InceptionFeatureExtractor,
    save_metrics_table,
)
from lip_sync.models.aux_renderer import AuxiliaryRenderer
from lip_sync.models.motion_gen import VisemeEncoder
from lip_sync.models.text_to_viseme import TextToVisemeProcessor
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
    lmd_sample_every_n: int = 1    # 每隔 N 个样本计算一次 LMD（1=全部计算）
    compute_syncnet: bool = True
    skip_generation: bool = False
    debug: bool = False


class EvaluationRunner:
    """评测运行器。"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu")
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

        pt_files = sorted(data_dir.glob(
            "*.pt")) or sorted(data_dir.rglob("*.pt"))
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
            model = VisemeEncoder(
                num_visemes=self.processor.vocab_size, d_model=512, out_dim=19200)
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
            self._inception_extractor = InceptionFeatureExtractor(
                device=self.device)
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
                    ids = self.processor.process(
                        text).unsqueeze(0).to(self.device)
                    target_T = ids.shape[1] * 3 if not cfg.use_duration else T
                    fau = gt.get("fau_signals")
                    if fau is not None and cfg.use_fau:
                        fau = fau.unsqueeze(0).to(self.device)
                        if fau.shape[1] != target_T:
                            fau = torch.nn.functional.interpolate(
                                fau.permute(0, 2, 1), size=target_T, mode='linear', align_corners=False
                            ).permute(0, 2, 1)
                    else:
                        fau = None
                    return model(ids, target_frames_len=target_T, fau_signals=fau)
                return fn

            results.update(self._run_fast_variant(
                name, samples, gt_frames_all, real_feats, make_abl_fn(
                    ablation_cfg, model_abl),
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
                # (1, T, 19200)
                feats = feature_fn(s, gt)
                feats = feats.to(self.device)                     # 确保在正确设备上
                # (T, 3, 64, 64)
                pred = self.aux_renderer(feats)[0]

                T = min(pred.shape[0], gt["gt_frames"].shape[0])
                gcrop, gt_crop = pred[:T], gt["gt_frames"][:T].to(pred.device)

                # 逐帧指标（每样本一个均值）
                p = calculate_psnr(gcrop.unsqueeze(
                    0), gt_crop.unsqueeze(0)).mean().item()
                s = calculate_ssim(gcrop.unsqueeze(
                    0), gt_crop.unsqueeze(0)).mean().item()
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

        metrics = {"FID": fid, "PSNR": float(
            np.mean(p_list)), "SSIM": float(np.mean(s_list))}
        if lm_gen:
            lmd, lmd_std, valid = calculate_lmd(lm_gen, lm_gt)
            metrics.update(
                {"LMD": lmd, "LMD_std": lmd_std, "ValidRatio": valid})

        print(
            f"  FID={fid:.4f}  PSNR={np.mean(p_list):.2f}  SSIM={np.mean(s_list):.4f}", end="")
        if "LMD" in metrics:
            print(f"  LMD={metrics['LMD']:.4f}", end="")
        print()

        return {name: metrics}

    # ── full 模式 ──

    def _evaluate_full_mode(self) -> Dict[str, Dict]:
        """
        完整模式：生成 512x512 扩散视频，计算所有指标（FID / PSNR / SSIM / LMD / SyncNet）。
        仅对比 Ours vs Audio-Driven 两个主方法。
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

        # ── 生成阶段 ──
        if not self.config.skip_generation:
            for method_key, _ in methods:
                print(f"\n[*] 正在生成 [{method_key}]...")
                for sidx, s in enumerate(samples):
                    if sidx % 5 == 0:
                        print(f"  [{sidx}/{len(samples)}] {s['name']}")
                    self._gen_video(s, method_key)

        # ── 加载全部 GT（保留逐样本数据用于逐帧指标）──
        print("\n[*] 加载真实帧...")
        gt_samples = []
        for s in samples:
            d = self._get_gt_data(s)
            gt_samples.append(d)
        gt_all = torch.cat([d["gt_frames"] for d in gt_samples], dim=0)

        print("[*] 计算真实帧 Inception 特征（用于 FID）...")
        real_feats = extract_inception_features(
            gt_all, self.inception_extractor, self.device, self.config.batch_size_fid
        )

        # ── 评测每个方法 ──
        for method_key, _ in methods:
            print(f"\n{'=' * 40}")
            print(f"评测 [{method_key}]")
            print(f"{'=' * 40}")

            method_dir = self.output_root / method_key
            if not method_dir.exists():
                print(f"  [-] 目录不存在，跳过")
                continue

            gen_frame_list = []      # for FID
            psnr_list = []
            ssim_list = []
            lm_gen_all = []
            lm_gt_all = []
            syncnet_list = []
            skipped = 0

            for sidx, s in enumerate(samples):
                vpath = method_dir / f"{s['name']}.mp4"
                if not vpath.exists() or vpath.stat().st_size <= 1024:
                    skipped += 1
                    continue

                # 加载生成帧和 GT 帧
                try:
                    gen_frames = self._load_video(str(vpath))  # (T, C, H, W), [-1, 1]
                except Exception as e:
                    print(f"  [-] 加载失败 {vpath.name}: {e}")
                    skipped += 1
                    continue

                gt_data = gt_samples[sidx]
                gt_frames = gt_data["gt_frames"]                 # (T_gt, C, H_gt, W_gt), [-1, 1]
                whisper = gt_data["whisper_features"]            # (T_whisper, 19200)

                # 对齐时间步
                T = min(gen_frames.shape[0], gt_frames.shape[0], whisper.shape[0])
                if T < 5:
                    print(f"  [-] 帧数不足 ({T})，跳过 {s['name']}")
                    skipped += 1
                    continue
                gen_crop = gen_frames[:T]                        # (T, C, H, W)
                gt_crop = gt_frames[:T]                          # (T, C, H_gt, W_gt)

                # ── FID 累积 ──
                gen_frame_list.append(gen_crop)

                # ── PSNR / SSIM（将生成帧缩放到 GT 分辨率）──
                if gen_crop.shape[-2:] != gt_crop.shape[-2:]:
                    gen_for_compare = torch.nn.functional.interpolate(
                        gen_crop.permute(1, 0, 2, 3).unsqueeze(0),  # (1, C, T, H, W)
                        size=(gt_crop.shape[-2], gt_crop.shape[-1]),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0).permute(1, 0, 2, 3)              # back to (T, C, H_gt, W_gt)
                else:
                    gen_for_compare = gen_crop

                # 注意：PSNR/SSIM 输入应为 (B, T, C, H, W)
                psnr_per_frame = calculate_psnr(
                    gen_for_compare.unsqueeze(0), gt_crop.unsqueeze(0)
                )  # (1, T)
                ssim_per_frame = calculate_ssim(
                    gen_for_compare.unsqueeze(0), gt_crop.unsqueeze(0)
                )  # (1, T)

                psnr_list.append(psnr_per_frame.mean().item())
                ssim_list.append(ssim_per_frame.mean().item())

                # ── LMD（每隔 lmd_sample_every_n 个样本计算一次）──
                if self.config.compute_lmd and sidx % self.config.lmd_sample_every_n == 0:
                    # 帧二次采样：每个样本最多处理 100 帧
                    step = max(1, T // 100)
                    lm_gen = extract_lip_landmarks(gen_crop[::step])
                    lm_gt = extract_lip_landmarks(gt_crop[::step])
                    lm_gen_all.extend(lm_gen)
                    lm_gt_all.extend(lm_gt)

                # ── SyncNet ──
                if self.config.compute_syncnet:
                    try:
                        sc, _ = calculate_syncnet_score(
                            gen_crop, whisper[:T], device=self.device
                        )
                        syncnet_list.append(sc)
                    except Exception as e:
                        print(f"  [-] SyncNet 失败 [{s['name']}]: {e}")

            if skipped:
                print(f"  [-] 跳过了 {skipped}/{len(samples)} 个无效样本")

            # ── 聚合 ──
            if len(gen_frame_list) < 2:
                print(f"  [-] 有效样本不足（<2），跳过 {method_key}")
                continue

            # FID
            total_frames = sum(f.shape[0] for f in gen_frame_list)
            print(f"  计算 FID ({total_frames} 帧)...")
            gen_all = torch.cat(gen_frame_list, dim=0)
            gen_feats = extract_inception_features(
                gen_all, self.inception_extractor, self.device, self.config.batch_size_fid
            )
            fid = calculate_fid(real_feats, gen_feats)

            metrics = {
                "FID": fid,
                "PSNR": float(np.mean(psnr_list)),
                "SSIM": float(np.mean(ssim_list)),
            }

            # LMD
            if lm_gen_all:
                lmd, lmd_std, valid_ratio = calculate_lmd(lm_gen_all, lm_gt_all)
                metrics.update({
                    "LMD": lmd,
                    "LMD_std": lmd_std,
                    "ValidRatio": valid_ratio,
                })

            # SyncNet
            if syncnet_list:
                metrics["SyncNet"] = float(np.mean(syncnet_list))

            results[method_key] = metrics

            # 打印摘要
            summary = f"  FID={fid:.4f}  PSNR={np.mean(psnr_list):.2f}  SSIM={np.mean(ssim_list):.4f}"
            if "LMD" in metrics:
                summary += f"  LMD={metrics['LMD']:.4f}"
            if "SyncNet" in metrics:
                summary += f"  SyncNet={metrics['SyncNet']:.4f}"
            print(summary)

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
            frames.append(torch.from_numpy(cv2.cvtColor(
                f, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 127.5 - 1.0)
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
            save_metrics_table(self.results, str(
                self.output_root / self.config.results_file))
            save_metrics_table(self.results, str(
                self.output_root / "results.md"), fmt="markdown")

        return self.results


def main():
    p = argparse.ArgumentParser(description="Text2Lip 评测脚本")
    p.add_argument("--mode", choices=["fast", "full"], default="fast",
                   help="fast=64x64辅助渲染器 | full=512x512全扩散")
    p.add_argument("--test-samples", type=int, default=None)
    p.add_argument("--test-split-ratio", type=float, default=0.1)
    p.add_argument("--eval-only", action="store_true", help="跳过生成，仅评测已有视频")
    p.add_argument("--data-dir", default="data/dataset/grid/preprocessed/s1")
    p.add_argument(
        "--weights", default="pretrained_weights/text_driven_model.pth")
    p.add_argument("--output", default="output/evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-lmd", action="store_true", help="跳过 LMD 计算（节省时间）")
    p.add_argument("--lmd-sample-every", type=int, default=1,
                   help="每隔 N 个样本计算一次 LMD（默认 1=全部）")
    p.add_argument("--no-syncnet", action="store_true", help="跳过 SyncNet 计算")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    cfg = EvalConfig(
        mode=args.mode, test_samples=args.test_samples,
        test_split_ratio=args.test_split_ratio, data_dir=args.data_dir,
        text_model_weights=args.weights, output_root=args.output,
        batch_size_fid=args.batch_size, compute_lmd=not args.no_lmd,
        lmd_sample_every_n=args.lmd_sample_every,
        compute_syncnet=not args.no_syncnet,
        skip_generation=args.eval_only, device=args.device,
        seed=args.seed, debug=args.debug,
    )
    EvaluationRunner(cfg).run_all()


if __name__ == "__main__":
    main()
