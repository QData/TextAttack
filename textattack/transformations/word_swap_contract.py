from textattack.transformations import Transformation


class WordSwapContract(Transformation):
    """Transforms an input by performing contraction on recognized
    combinations."""

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
        """Return all possible transformed sentences, each with one
        contraction."""
        transformed_texts = []

        words = current_text.words
        indices_to_modify = list(indices_to_modify)

        # search for every 2-words combination in reverse_contraction_map
        for idx in indices_to_modify[:-1]:
            word = words[idx]

            next_idx = indices_to_modify[indices_to_modify.index(idx) + 1]
            next_word = words[next_idx]

            # generating the words to search for
            key = " ".join([word, next_word])

            # when a possible contraction is found in map, contract the current text
            if key in self.reverse_contraction_map:
                transformed_text = current_text.replace_word_at_index(
                    idx, self.reverse_contraction_map[key]
                )
                transformed_text = transformed_text.delete_word_at_index(next_idx)
                transformed_texts.append(transformed_text)

        return transformed_texts
