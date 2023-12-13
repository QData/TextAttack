from collections import defaultdict

from textattack.constraints import PreTransformationConstraint


class UnmodifablePhrases(PreTransformationConstraint):
    """A constraint that prevents the modification of specified phrases or
    words.

    Args:
        phrases (list(str)): A list of strings that cannot be modified
    """

    def __init__(self, phrases):
        self.length_to_phrases = defaultdict(set)
        for phrase in phrases:
            self.length_to_phrases[len(phrase.split())].add(phrase.lower())

    def _get_modifiable_indices(self, current_text):
        phrase_indices = set()

        for phrase_length in self.length_to_phrases.keys():
            for i in range(len(current_text.words) - phrase_length + 1):
                if (
                    " ".join(current_text.words[i : i + phrase_length])
                    in self.length_to_phrases[phrase_length]
                ):
                    phrase_indices |= set(range(i, i + phrase_length))

        return set(i for i in range(len(current_text.words)) if i not in phrase_indices)

    def extra_repr_keys(self):
        return ["phrases"]
