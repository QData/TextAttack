from .utils import get_device

class TokenizedText:
    def __init__(self, model, text):
        """ Initializer stores text and tensor of tokenized text.
        """
        self.model = model
        self.text = text
        self.ids = model.convert_text_to_ids(text)
        self.raw_words = raw_words(text)
    
    def words(self):
        """ Returns the distinct words from self.text. 
        
            @TODO Should we ever consider case when substituting words?
        """
        return self.raw_words
    
    def text_until_word_index(self, i):
        """ Returns the text before the beginning of word at index `i`. """
        pre_words = self.words()[:i+1]
        lower_text = self.text.lower()
        # Find all words until `i` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word, look_after_index)
        return self.text[:look_after_index]
    
    def text_after_word_index(self, i):
        """ Returns the text before the beginning of word at index `i`. """
        word = self.words()[i]
        pre_words = self.words()[:i+1]
        lower_text = self.text.lower()
        # Find all words until `i` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word, look_after_index)
        # Jump to the end of word.
        look_after_index += len(word)
        return self.text[look_after_index:]
    
    def first_word_diff(self, other_tokenized_text):
        """ Returns the first word in self.raw_words() that differs from 
            other_tokenized_text. Useful for word swap strategies. """
        w1 = self.raw_words
        w2 = other_tokenized_text.words()
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return w1
        return None
    
    def first_word_diff_index(self, other_tokenized_text):
        """ Returns the index of the first word in self.raw_words() that differs
            from other_tokenized_text. Useful for word swap strategies. """
        w1 = self.raw_words
        w2 = other_tokenized_text.words()
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return i
        return None
   
    def all_words_diff(self, other_tokenized_text):
        """ Returns the set of indices for which this and other_tokenized_text
        have different words. """
        indices = set()
        w1 = self.raw_words
        w2 = other_tokenized_text.words()
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                indices.add(i)
        return indices
        
    def ith_word_diff(self, other_tokenized_text, i):
        """ Returns whether the word at index i differs from other_tokenized_text
        """
        w1 = self.raw_words
        w2 = other_tokenized_text.words()
        if len(w1) - 1 < i or len(w2) - 1 < i:
            return True
        return w1[i] != w2[i]

    def replace_words_at_indices(self, indices, words):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        if len(indices) != len(words):
            raise ValueError(f'Cannot replace {len(words)} words at {len(indices)} indices.')
        new_words = self.raw_words[:]
        for i, word in zip(indices, words):
            new_words[i] = word
        return self.replace_new_words(new_words)
    
    def replace_word_at_index(self, index, new_word):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        return self.replace_words_at_indices([index], [new_word])
    
    def replace_new_words(self, new_words):
        """ This code returns a new TokenizedText object and replaces old list 
            of words with a new list of words, but preserves the punctuation 
            and spacing of the original message.
        """
        final_sentence = ''
        text = self.text
        for input_word, adv_word in zip(self.raw_words, new_words):
            if input_word == '[UNKNOWN]': continue
            word_start = text.index(input_word)
            word_end = word_start + len(input_word)
            final_sentence += text[:word_start]
            final_sentence += adv_word
            text = text[word_end:]
        final_sentence += text # Add all of the ending punctuation.
        return TokenizedText(self.model, final_sentence)
        
    def __repr__(self):
        return self.text

def raw_words(s):
    """ Lowercases a string, removes all non-alphanumeric characters,
        and splits into words. """
    words = []
    word = ''
    for c in ' '.join(s.split()):
        if c.isalpha():
            word += c
        elif word:
            words.append(word)
            word = ''
    if word: words.append(word)
    return words
