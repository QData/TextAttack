import re

from textattack.shared import AttackedText
from textattack.transformations import Transformation


class WordSwapContract(Transformation):

    reverse_contraction_map = {
        "is not": "isn't",
        "are not": "aren't",
        "cannot": "can't",
        "could not": "couldn't",
        "did not": "didn't",
        "does not": "doesn't",
        "do not": "don't",
        "had not": "hadn't",
        "has not": "hasn't",
        "have not": "haven't",
        "he is": "he's",
        "how did": "how'd",
        "how is": "how's",
        "I would": "I'd",
        "I will": "I'll",
        "I am": "I'm",
        "i would": "i'd",
        "i will": "i'll",
        "i am": "i'm",
        "it would": "it'd",
        "it will": "it'll",
        "it is": "it's",
        "might not": "mightn't",
        "must not": "mustn't",
        "need not": "needn't",
        "ought not": "oughtn't",
        "shall not": "shan't",
        "she would": "she'd",
        "she will": "she'll",
        "she is": "she's",
        "should not": "shouldn't",
        "that would": "that'd",
        "that is": "that's",
        "there would": "there'd",
        "there is": "there's",
        "they would": "they'd",
        "they will": "they'll",
        "they are": "they're",
        "was not": "wasn't",
        "we would": "we'd",
        "we will": "we'll",
        "we are": "we're",
        "were not": "weren't",
        "what are": "what're",
        "what is": "what's",
        "when is": "when's",
        "where did": "where'd",
        "where is": "where's",
        "who will": "who'll",
        "who is": "who's",
        "who have": "who've",
        "why is": "why's",
        "will not": "won't",
        "would not": "wouldn't",
        "you would": "you'd",
        "you will": "you'll",
        "you are": "you're",
    }

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        words = current_text.words
        indices_to_modify = list(indices_to_modify)

        for idx in indices_to_modify:
            word = words[idx]

            try:
                next_idx = indices_to_modify[indices_to_modify.index(idx) + 1]
                next_word = words[next_idx]
            except IndexError:
                continue

            key = " ".join([word, next_word])
            if key in self.reverse_contraction_map:
                transformed_text = current_text.replace_word_at_index(
                    idx, self.reverse_contraction_map[key]
                )
                transformed_text = transformed_text.delete_word_at_index(next_idx)
                transformed_texts.append(transformed_text)

        return transformed_texts
