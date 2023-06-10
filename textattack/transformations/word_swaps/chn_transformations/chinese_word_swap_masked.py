"""
Word Swap by BERT-Masked LM.
-------------------------------
"""

from transformers import pipeline

from . import WordSwap


class ChineseWordSwapMaskedLM(WordSwap):
    """Generate potential replacements for a word using a masked language
    model."""

    def __init__(self, task="fill-mask", model="xlm-roberta-base", **kwargs):
        self.unmasker = pipeline(task, model)
        super().__init__(**kwargs)

    def get_replacement_words(self, current_text, indice_to_modify):
        masked_text = current_text.replace_word_at_index(indice_to_modify, "<mask>")
        outputs = self.unmasker(masked_text.text)
        words = []
        for dict in outputs:
            take = True
            for char in dict["token_str"]:
                # accept only Chinese characters for potential substitutions
                if not is_cjk(char):
                    take = False
            if take:
                words.append(dict["token_str"])

        return words

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
