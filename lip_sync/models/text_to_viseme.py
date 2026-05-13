import re
import torch
from g2p_en import G2p
import nltk

# 每帧时长 = 1000ms / 25fps = 40ms
FPS = 25
MS_PER_FRAME = 1000.0 / FPS  # 40ms

class TextToVisemeProcessor:
    def __init__(self, fps=FPS):
        # 确保下载必要的 nltk 数据
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            
        self.g2p = G2p()
        self.fps = fps

        # 预定义的 音素-视素 映射表
        self.phoneme_to_viseme = {
            'SIL': 0, 'SP': 0,
            'AA': 1, 'AO': 1, 'AH': 1,
            'AE': 2, 'EH': 2, 'IH': 2, 'IY': 2,
            'AW': 3, 'OW': 3, 'UW': 3, 'UH': 3,
            'ER': 4,
            'P': 5, 'B': 5, 'M': 5,
            'F': 6, 'V': 6,
            'T': 7, 'D': 7, 'S': 7, 'Z': 7,
            'TH': 8, 'DH': 8,
            'SH': 9, 'ZH': 9, 'CH': 9, 'JH': 9,
            'K': 10, 'G': 10, 'N': 10, 'NG': 10,
            'L': 11,
            'R': 12,
            'Y': 13,
            'W': 14,
            'HH': 15
        }
        self.vocab_size = max(self.phoneme_to_viseme.values()) + 1

        # ============================================================
        # 音素级时长映射表（基于英语语言学统计）
        # 单位：帧（1帧 = 40ms @ 25fps）
        #
        # 参考来源：英语语音学中不同音素类别的典型时长范围
        #   - 短元音 (IH, EH, AE, AH, UH):  ~80-120ms  → 2-3帧
        #   - 长元音 (IY, AA, AO, ER, OW, UW): ~150-200ms → 4-5帧
        #   - 双元音 (AW, AY, EY, OY):        ~180-240ms → 5-6帧
        #   - 清/浊擦音 (F, V, TH, DH, S, Z...): ~80-140ms → 2-4帧
        #   - 爆破音 (P, B, T, D, K, G):      ~40-80ms   → 1-2帧
        #   - 塞擦音 (CH, JH):                ~80-120ms  → 2-3帧
        #   - 鼻音 (M, N, NG):                ~80-120ms  → 2-3帧
        #   - 流音 (L, R):                    ~60-120ms  → 2-3帧
        #   - 滑音 (Y, W):                    ~60-100ms  → 2-3帧
        #   - 声门音 (HH):                    ~60-100ms  → 2-3帧
        #   - 停顿 (空格/标点):               ~80-160ms  → 2-4帧
        # ============================================================
        self.phoneme_durations = {
            # Silence / pauses
            'SIL': 2, 'SP': 2,
            # Short vowels (~2-3 frames)
            'IH': 3, 'EH': 3, 'AE': 3, 'AH': 3, 'UH': 3,
            # Long vowels (~4-5 frames)
            'IY': 5, 'AA': 5, 'AO': 5, 'ER': 5, 'OW': 5, 'UW': 5,
            # Diphthongs (~5-6 frames)
            'AW': 6, 'AY': 6, 'EY': 6, 'OY': 6,
            # Fricatives (~2-4 frames)
            'F': 3, 'V': 3, 'TH': 3, 'DH': 3, 'S': 3, 'Z': 3, 'SH': 3, 'ZH': 3, 'HH': 3,
            # Plosives (~1-2 frames)
            'P': 2, 'B': 2, 'T': 2, 'D': 2, 'K': 2, 'G': 2,
            # Affricates (~2-3 frames)
            'CH': 3, 'JH': 3,
            # Nasals (~2-3 frames)
            'M': 3, 'N': 3, 'NG': 3,
            # Liquids (~2-3 frames)
            'L': 3, 'R': 3,
            # Glides (~2 frames)
            'Y': 2, 'W': 2,
        }

    def process(self, text):
        """
        将输入文本转化为视素ID序列（保持原有接口，仅返回视素ID）
        """
        phonemes = self.g2p(text)
        viseme_seq = []
        for p in phonemes:
            if p in [" ", ".", ",", "?", "!"]:
                viseme_seq.append(0)
                continue
            clean_p = re.sub(r'\d+', '', p)
            viseme_id = self.phoneme_to_viseme.get(clean_p, 0)
            viseme_seq.append(viseme_id)
        return torch.tensor(viseme_seq, dtype=torch.long)

    def process_with_duration(self, text):
        """
        将输入文本转化为「展开后的」视素ID序列 + 总帧数
        - 先通过 G2P 获取音素序列
        - 基于音素时长映射表，为每个音素分配帧数
        - 按帧数展开（重复）每个视素ID，得到展开后的视素序列
        - 返回：展开后的视素序列 (torch.long), 总帧数 (int)
        
        示例:
          输入: "I am" → 音素: AY(6帧) SP(2帧) AE(3帧) M(3帧)
          展开: [13,13,13,13,13,13, 0,0, 2,2,2, 5,5,5]
          总帧数: 14
        """
        phonemes = self.g2p(text)
        expanded_visemes = []
        frame_counts = []  # 每个音素对应的帧数（用于调试）
        
        for p in phonemes:
            # 处理标点/空格 → 映射为停顿
            if p in [" ", ".", ",", "?", "!"]:
                # 标点符号的停顿时长
                pause_frames = {
                    ' ': 2,
                    '.': 4,
                    ',': 2,
                    '?': 4,
                    '!': 4,
                }.get(p, 2)
                expanded_visemes.extend([0] * pause_frames)
                frame_counts.append(pause_frames)
                continue
            
            # 提取干净的音素（去掉重音标记数字）
            clean_p = re.sub(r'\d+', '', p)
            
            # 获取视素ID
            viseme_id = self.phoneme_to_viseme.get(clean_p, 0)
            
            # 获取该音素的时长（帧数）
            duration = self.phoneme_durations.get(clean_p, 3)  # 默认3帧
            if duration < 1:
                duration = 1  # 最少1帧
            
            # 按时长展开
            expanded_visemes.extend([viseme_id] * duration)
            frame_counts.append(duration)
        
        total_frames = len(expanded_visemes)
        
        return (
            torch.tensor(expanded_visemes, dtype=torch.long),
            total_frames,
            frame_counts,
            phonemes,
        )

if __name__ == "__main__":
    processor = TextToVisemeProcessor()
    text = "I am a text-driven talking head. No audio needed."
    
    # 原有接口（保持兼容）
    visemes = processor.process(text)
    print(f"Legacy - Viseme Sequence: {visemes}")
    print(f"Legacy - Length: {visemes.shape[0]} * 8 = {visemes.shape[0] * 8} frames\n")
    
    # 新接口（带时长预测）
    expanded, total_frames, frame_counts, phonemes = processor.process_with_duration(text)
    print(f"New - Input Text: '{text}'")
    print(f"New - Phonemes: {phonemes}")
    print(f"New - Frame counts per phoneme: {frame_counts}")
    print(f"New - Expanded viseme length: {expanded.shape[0]}")
    print(f"New - Total frames: {total_frames}")
    print(f"New - Estimated duration: {total_frames / FPS:.1f}s (at {FPS}fps)")
    print(f"New - Old method would produce: {visemes.shape[0] * 8 / FPS:.1f}s")
