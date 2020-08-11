import re

from textattack.transformations import WordSwap


class WordSwap_CE(WordSwap):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.contraction_map = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have", "couldn't":
                "could not", "didn't": "did not", "doesn't": "does not", "don't":
                "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
                "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
                "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'll": "I will", "I'm": "I am",
            "I've": "I have", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
                "madam", "might've": "might have", "mightn't": "might not",
            "must've": "must have", "mustn't": "must not", "needn't":
                "need not", "oughtn't": "ought not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "that'd":
                "that would", "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what're": "what are", "what's": "what is",
            "when's": "when is", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who's": "who is",
            "who've": "who have", "why's": "why is", "won't": "will not",
            "would've": "would have", "wouldn't": "would not",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }

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

    def expand_contractions(self, sentence, **kwargs):
        # self.reverse_contraction_map = dict([(y, x) for x, y in self.contraction_map.items()])
        contraction_pattern = re.compile(r'\b({})\b'.format('|'.join(self.contraction_map.keys())),
                                         flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            # print(0,contraction)
            match = contraction.group(0)
            # print(match)
            first_char = match[0]
            print(first_char)
            expanded_contraction = self.contraction_map.get(match, self.contraction_map.get(match.lower()))
            print(expanded_contraction)
            expanded_contraction = first_char + expanded_contraction[1:]
            # print(expanded_contraction)
            return expanded_contraction

        return contraction_pattern.sub(expand_match, sentence)

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

    def _get_replacement_words(self, word):
        output = []
        output.append(self.expand_contractions(word))
        output.append(self.contract(word))
        return output
