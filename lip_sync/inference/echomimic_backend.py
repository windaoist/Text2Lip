import os
import sys

# 将项目根目录和 EchoMimic 添加到 Python 路径
# 必须在导入 src.xxx 之前完成
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ECHOMIMIC_ROOT = os.path.join(PROJECT_ROOT, "third_party", "EchoMimic")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if ECHOMIMIC_ROOT not in sys.path:
    sys.path.insert(0, ECHOMIMIC_ROOT)  # 优先使用 EchoMimic 的 src

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.models.face_locator import FaceLocator
from src.utils.util import save_videos_grid, crop_and_pad
from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from pathlib import Path

# ==============================================================================
# 解决 diffusers/peft 与不同版本 huggingface_hub/transformers 的兼容性问题
# ==============================================================================
try:
    import huggingface_hub.errors
    if not hasattr(huggingface_hub.errors, 'LocalEntryNotFoundError'):
        class LocalEntryNotFoundError(Exception):
            pass
        huggingface_hub.errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    if not hasattr(huggingface_hub.errors, 'EntryNotFoundError'):
        class EntryNotFoundError(Exception):
            pass
        huggingface_hub.errors.EntryNotFoundError = EntryNotFoundError
except ImportError:
    pass

try:
    import transformers
    if not hasattr(transformers, 'EncoderDecoderCache'):
        class EncoderDecoderCache:
            pass
        transformers.EncoderDecoderCache = EncoderDecoderCache
except ImportError:
    pass


class EchoMimicBackend:
    def __init__(self, config_path=None, device="cuda"):
        if config_path is None:
            config_path = os.path.join(
                ECHOMIMIC_ROOT, "configs", "prompts", "animation.yaml")

        self.config = OmegaConf.load(config_path)
        # 修正配置中的路径为绝对路径
        self._fix_paths()

        self.device = device if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if self.config.weight_dtype == "fp16" else torch.float32

        self.infer_config = OmegaConf.load(self.config.inference_config)
        self.initialize_models()

    def _fix_paths(self):
        """将配置文件中的相对路径转换为绝对路径"""
        # 模型文件通常在项目根目录的 pretrained_weights 下
        model_keys = [
            "pretrained_base_model_path", "pretrained_vae_path", "audio_model_path",
            "denoising_unet_path", "reference_unet_path", "face_locator_path",
            "motion_module_path"
        ]
        for key in model_keys:
            if key in self.config and self.config[key].startswith("./"):
                self.config[key] = os.path.join(
                    PROJECT_ROOT, self.config[key][2:])

        # 配置文件 (如 inference_config) 通常在 EchoMimic 目录下
        if "inference_config" in self.config and self.config["inference_config"].startswith("./"):
            self.config["inference_config"] = os.path.join(
                ECHOMIMIC_ROOT, self.config["inference_config"][2:])

    def initialize_models(self):
        import gc
        print("[*] 正在初始化 EchoMimic 模型 (已开启低内存优化)...")

        # 强制 PyTorch 默认在 CPU 上使用 float16 初始化张量，极大降低系统 RAM 占用
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.weight_dtype)

        try:
            # 1. VAE
            self.vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_vae_path,
                torch_dtype=self.weight_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            gc.collect()

            # 2. Reference UNet
            self.reference_unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_base_model_path,
                subfolder="unet",
                torch_dtype=self.weight_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)

            # 使用 mmap=True (内存映射) 零拷贝加载权重，防止 CPU RAM 飙升
            ref_sd = torch.load(self.config.reference_unet_path,
                                map_location="cpu", mmap=True)
            self.reference_unet.load_state_dict(ref_sd)
            del ref_sd
            gc.collect()

            # 3. Denoising UNet (最耗内存的部分)
            if os.path.exists(self.config.motion_module_path):
                self.denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                    self.config.pretrained_base_model_path,
                    self.config.motion_module_path,
                    subfolder="unet",
                    unet_additional_kwargs=self.infer_config.unet_additional_kwargs,
                ).to(dtype=self.weight_dtype, device=self.device)
            else:
                self.denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
                    self.config.pretrained_base_model_path,
                    "",
                    subfolder="unet",
                    unet_additional_kwargs={
                        "use_motion_module": False,
                        "unet_use_temporal_attention": False,
                        "cross_attention_dim": self.infer_config.unet_additional_kwargs.cross_attention_dim
                    }
                ).to(dtype=self.weight_dtype, device=self.device)

            denoise_sd = torch.load(
                self.config.denoising_unet_path, map_location="cpu", mmap=True)
            self.denoising_unet.load_state_dict(denoise_sd, strict=False)
            del denoise_sd
            gc.collect()

            # 4. Face Locator
            self.face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(
                dtype=self.weight_dtype, device=self.device
            )
            locator_sd = torch.load(
                self.config.face_locator_path, map_location="cpu", mmap=True)
            self.face_locator.load_state_dict(locator_sd)
            del locator_sd
            gc.collect()

        finally:
            # 恢复 PyTorch 的默认数据类型，避免影响其他常规计算
            torch.set_default_dtype(original_dtype)

        # 5. Audio Processor
        if os.path.exists(self.config.audio_model_path):
            self.audio_processor = load_audio_model(
                model_path=self.config.audio_model_path, device=self.device)
        else:
            print(
                f"[-] 警告: 未找到音频模型 {self.config.audio_model_path}，将跳过加载。纯文本驱动模式不受影响。")
            self.audio_processor = None
        gc.collect()

        # 6. Face Detector
        self.face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                                   factor=0.709, post_process=True, device=self.device)
        gc.collect()

        # 7. Pipeline
        sched_kwargs = OmegaConf.to_container(
            self.infer_config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.pipe = Audio2VideoPipeline(
            vae=self.vae,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            audio_guider=self.audio_processor,
            face_locator=self.face_locator,
            scheduler=self.scheduler,
        ).to(self.device, dtype=self.weight_dtype)

        torch.cuda.empty_cache()
        print("[*] EchoMimic 初始化完成。")

    def load_text_driven_model(self, model_path="pretrained_weights/text_driven_model.pth"):
        """加载训练好的文本驱动模型 (更新为直接输出 19200 维特征)"""
        from lip_sync.models.motion_gen import VisemeEncoder
        from lip_sync.models.text_to_viseme import TextToVisemeProcessor
        processor = TextToVisemeProcessor()
        
        # 目标维度 19200 (EchoMimic Whisper Feature Chunk Size: 50 * 384 = 19200)
        self.text_model = VisemeEncoder(
            num_visemes=processor.vocab_size, d_model=512, out_dim=19200).to(self.device)
        
        if os.path.exists(model_path):
            try:
                self.text_model.load_state_dict(
                    torch.load(model_path, map_location=self.device))
                self.text_model.eval()
                print(f"[*] 已加载文本驱动模型: {model_path}")
            except Exception as e:
                print(f"[-] 加载权重失败: {e}")
        else:
            print(f"[-] 警告: 未找到文本驱动模型 {model_path}，将使用随机初始化。")

    def generate_from_text(self, ref_image_path, viseme_ids, output_path, target_frames_len=None, width=512, height=512, steps=30, cfg=2.5, fps=24, seed=420, progress_callback=None):
        """
        直接从视素 ID 序列生成视频
        支持 progress_callback(percent, stage, message) 用于实时进度反馈
        """
        if not hasattr(self, 'text_model'):
            self.load_text_driven_model()

        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(np.random.randint(100, 1000000))

        # 1. 预处理参考图像 (人脸检测与裁剪)
        if progress_callback:
            progress_callback(0, 'preprocess', '正在预处理参考图像...')
        
        face_img = cv2.imread(ref_image_path)
        face_mask = np.zeros(
            (face_img.shape[0], face_img.shape[1])).astype('uint8')
        det_bboxes, probs = self.face_detector.detect(face_img)
        select_bbox = self._select_face(det_bboxes, probs)

        if select_bbox is None:
            face_mask[:, :] = 255
        else:
            xyxy = np.round(select_bbox[:4]).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * 0.1)
            c_pad = int((ce - cb) * 0.1)
            face_mask[max(0, rb - r_pad): min(face_img.shape[0], re + r_pad),
                      max(0, cb - c_pad): min(face_img.shape[1], ce + c_pad)] = 255

            # face crop
            r_pad_crop = int((re - rb) * 0.5)
            c_pad_crop = int((ce - cb) * 0.5)
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop),
                         min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
            face_img, _ = crop_and_pad(face_img, crop_rect)
            face_mask, _ = crop_and_pad(face_mask, crop_rect)
            face_img = cv2.resize(face_img, (width, height))
            face_mask = cv2.resize(face_mask, (width, height))

        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        face_mask_tensor = torch.Tensor(face_mask).to(
            dtype=self.weight_dtype, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            # 2. 文本 -> 视素 -> 特征
            if viseme_ids.dim() == 1:
                viseme_ids = viseme_ids.unsqueeze(0) # (1, N)
            viseme_ids = viseme_ids.to(self.device)
            
            if target_frames_len is None:
                target_frames_len = viseme_ids.shape[1] * 8
            
            if progress_callback:
                progress_callback(5, 'text_features', '正在提取文本特征...')
                
            text_features = self.text_model(viseme_ids, target_frames_len=target_frames_len)  # (1, T, 19200)
            
            if progress_callback:
                progress_callback(10, 'text_features', '文本特征提取完成，准备生成视频...')

            # 3. 映射到 EchoMimic 期望的维度 (1, T, 50, 384)
            T = text_features.shape[1]
            audio_fea_final = text_features.view(1, T, 50, 384).to(dtype=self.weight_dtype)

            # 4. 自定义 pipeline callback 来跟踪扩散步骤进度和帧生成进度
            diffusion_total = steps  # e.g. 30
            frames_total = T       # e.g. 132
            # 用于帧解码进度的可修改容器 (中文注释)
            frame_decode_count = [0]
            # 扩散进度百分比的上限: 扩散结束后进度为70% (中文注释)
            DIFFUSION_END_PERCENT = 70
            
            # 扩散步骤回调 (10% ~ 70%)
            def make_pipeline_callback():
                pipe_step_count = [0]
                
                def callback(pipe, step_index, timestep, callback_kwargs):
                    nonlocal pipe_step_count
                    pipe_step_count[0] += 1
                    
                    # 扩散步骤进度 (10% ~ 70%)
                    diffusion_progress = (pipe_step_count[0] / diffusion_total) * (DIFFUSION_END_PERCENT - 10)
                    overall = 10 + diffusion_progress
                    
                    msg = f'扩散步骤 {pipe_step_count[0]}/{diffusion_total}'
                    if progress_callback:
                        progress_callback(overall, 'diffusion', msg)
                    
                    return callback_kwargs
                return callback
            
            # 帧解码回调 (70% ~ 95%) 由 pipeline 的 decode_latents 逐帧调用 (中文注释)
            def frame_decode_callback(frame_idx, total_frames):
                frame_decode_count[0] = frame_idx + 1
                # 帧解码进度: 70% ~ 95%
                frame_progress = (frame_decode_count[0] / total_frames) * (95 - DIFFUSION_END_PERCENT)
                overall = DIFFUSION_END_PERCENT + frame_progress
                msg = f'正在生成视频帧 {frame_decode_count[0]}/{total_frames}'
                if progress_callback:
                    progress_callback(overall, 'frame', msg)
            
            if progress_callback:
                progress_callback(10, 'diffusion_start', f'开始扩散生成 ({diffusion_total} 步, {T} 帧)...')
            
            print(f"[*] 正在从文本特征生成视频 (帧数: {T})...")

            video = self.pipe(
                ref_image_pil,
                None,  # audio_path 为空
                face_mask_tensor,
                width,
                height,
                T,
                steps,
                cfg,
                generator=generator,
                audio_sample_rate=16000,
                context_frames=12,
                fps=fps,
                context_overlap=3,
                audio_fea_final=audio_fea_final,  # 传入预计算的特征
                callback=make_pipeline_callback() if progress_callback else None,
                callback_steps=1,
                frame_callback=frame_decode_callback if progress_callback else None,  # 帧解码进度回调 (中文注释)
            ).videos

            if progress_callback:
                progress_callback(95, 'saving', '正在保存视频文件...')

            final_video = torch.cat([video], dim=0)
            save_videos_grid(final_video, output_path, fps=fps)

            print(f"[*] 文本驱动视频生成完成: {output_path}")
            
            if progress_callback:
                progress_callback(100, 'complete', '视频生成完成!')
                
            return output_path

    def _select_face(self, det_bboxes, probs):
        if det_bboxes is None or probs is None:
            return None
        filtered_bboxes = []
        for bbox_i in range(len(det_bboxes)):
            if probs[bbox_i] > 0.8:
                filtered_bboxes.append(det_bboxes[bbox_i])
        if len(filtered_bboxes) == 0:
            return None
        sorted_bboxes = sorted(filtered_bboxes, key=lambda x: (
            x[3]-x[1]) * (x[2] - x[0]), reverse=True)
        return sorted_bboxes[0]

    def generate(self, ref_image_path, audio_path, output_path, width=512, height=512, length=1200, steps=30, cfg=2.5, fps=24, seed=420):
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(np.random.randint(100, 1000000))

        # Face detection and masking
        face_img = cv2.imread(ref_image_path)
        face_mask = np.zeros(
            (face_img.shape[0], face_img.shape[1])).astype('uint8')
        det_bboxes, probs = self.face_detector.detect(face_img)
        select_bbox = self._select_face(det_bboxes, probs)

        if select_bbox is None:
            face_mask[:, :] = 255
        else:
            xyxy = np.round(select_bbox[:4]).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * 0.1)
            c_pad = int((ce - cb) * 0.1)
            face_mask[max(0, rb - r_pad): min(face_img.shape[0], re + r_pad),
                      max(0, cb - c_pad): min(face_img.shape[1], ce + c_pad)] = 255

            # face crop
            r_pad_crop = int((re - rb) * 0.5)
            c_pad_crop = int((ce - cb) * 0.5)
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop),
                         min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
            face_img, _ = crop_and_pad(face_img, crop_rect)
            face_mask, _ = crop_and_pad(face_mask, crop_rect)
            face_img = cv2.resize(face_img, (width, height))
            face_mask = cv2.resize(face_mask, (width, height))

        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        face_mask_tensor = torch.Tensor(face_mask).to(
            dtype=self.weight_dtype, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0

        print(f"[*] 正在生成视频 (长度: {length} 帧)...")
        video = self.pipe(
            ref_image_pil,
            audio_path,
            face_mask_tensor,
            width,
            height,
            length,
            steps,
            cfg,
            generator=generator,
            audio_sample_rate=16000,
            context_frames=12,
            fps=fps,
            context_overlap=3
        ).videos

        final_video = torch.cat([video], dim=0)
        save_videos_grid(final_video, output_path, fps=fps)

        # 合并音频
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(output_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        output_with_audio = output_path.replace(".mp4", "_with_audio.mp4")
        final_clip.write_videofile(
            output_with_audio, codec="libx264", audio_codec="aac")

        print(f"[*] 视频生成完成: {output_with_audio}")
        return output_with_audio


if __name__ == "__main__":
    backend = EchoMimicBackend()
    # backend.generate(...)
