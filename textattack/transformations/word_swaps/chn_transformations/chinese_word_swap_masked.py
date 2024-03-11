"""
Word Swap by chinese BERT-Masked LM.
-------------------------------------
"""

import torch

from . import WordSwap


class ChineseWordSwapMaskedLM(WordSwap):
    """Generate potential replacements for a word using a masked language
    model."""

    def __init__(self, task="fill-mask", model="xlm-roberta-base", **kwargs):
        from transformers import BertForMaskedLM, BertTokenizer

        self.tt = BertTokenizer.from_pretrained(model)
        self.mm = BertForMaskedLM.from_pretrained(model)
        self.mm.to("cuda")
        super().__init__(**kwargs)

    def get_replacement_words(self, current_text, indice_to_modify):
        masked_text = current_text.replace_word_at_index(
            indice_to_modify, "[MASK]"
        )  # 修改前<mask>，xlmrberta的模型
        tokens = self.tt.tokenize(masked_text.text)
        input_ids = self.tt.convert_tokens_to_ids(tokens)
        input_tensor = torch.tensor([input_ids]).to("cuda")
        with torch.no_grad():
            outputs = self.mm(input_tensor)
            predictions = outputs.logits
        predicted_token_ids = torch.argsort(
            predictions[0, indice_to_modify], descending=True
        )[:50]
        predicted_tokens = self.tt.convert_ids_to_tokens(
            predicted_token_ids.tolist()[1:]
        )
        return predicted_tokens

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self.get_replacement_words(current_text, i)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts


def is_cjk(char):
    char = ord(char)
    for bottom, top in cjk_ranges:
        if bottom <= char <= top:
            return True
    return False


cjk_ranges = [
    (0x4E00, 0x62FF),
    (0x6300, 0x77FF),
    (0x7800, 0x8CFF),
    (0x8D00, 0x9FCC),
    (0x3400, 0x4DB5),
    (0x20000, 0x215FF),
    (0x21600, 0x230FF),
    (0x23100, 0x245FF),
    (0x24600, 0x260FF),
    (0x26100, 0x275FF),
    (0x27600, 0x290FF),
    (0x29100, 0x2A6DF),
    (0x2A700, 0x2B734),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x2F800, 0x2FA1F),
]
