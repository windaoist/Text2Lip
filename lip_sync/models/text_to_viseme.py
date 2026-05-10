import re
import torch
from g2p_en import G2p
import nltk

class TextToVisemeProcessor:
    def __init__(self):
        # 确保下载必要的 nltk 数据
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            nltk.download('averaged_perceptron_tagger_eng')
            
        self.g2p = G2p()

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

    def process(self, text):
        """将输入文本转化为视素ID序列"""
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

if __name__ == "__main__":
    processor = TextToVisemeProcessor()
    text = "Hello world!"
    visemes = processor.process(text)
    print(f"Step 1 - Input Text: '{text}'")
    print(f"Step 1 - Viseme Sequence: {visemes}")
