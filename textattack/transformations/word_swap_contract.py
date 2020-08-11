import re

from textattack.shared import AttackedText
from textattack.transformations import Transformation


class WordSwap_Contract(Transformation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reverse_contraction_map = {
            'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
            'could not': "couldn't", 'did not': "didn't", 'does not':
                "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
                "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
                "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
            'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
            'it would': "it'd", 'it will': "it'll", 'it is': "it's",
            'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
            'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
            'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
            'that would': "that'd", 'that is': "that's", 'there would':
                "there'd", 'there is': "there's", 'they would': "they'd",
            'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
            'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
                "weren't", 'what are': "what're", 'what is': "what's", 'when is':
                "when's", 'where did': "where'd", 'where is': "where's",
            'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
                "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
                "you'd", 'you will': "you'll", 'you are': "you're",
        }

    def contract(self, sentence, **kwargs):
        reverse_contraction_pattern = re.compile(r'\b({})\b '.format('|'.join(self.reverse_contraction_map.keys())),
                                                 flags=re.IGNORECASE | re.DOTALL)

        def cont(possible):
            match = possible.group(1)
            first_char = match[0]
            expanded_contraction = self.reverse_contraction_map.get(match,
                                                                    self.reverse_contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:] + ' '
            return expanded_contraction

        return reverse_contraction_pattern.sub(cont, sentence)

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        words = current_text.words
        words = " ".join(words)
        transformed_texts.append(AttackedText(self.contract(words)))
        return transformed_texts
