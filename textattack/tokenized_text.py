from utils import get_device

class TokenizedText:
    def __init__(self, model, text, modified_index=None):
        """ Initializer stores text and tensor of tokenized text.
        """
        self.model = model
        self.text = text
        self.ids = model.convert_text_to_ids(text)
        self.raw_words = raw_words(text)
        self.attack_attrs = dict()
        if modified_index:
            self.attack_attrs['modified_index'] = modified_index
    
    def words(self):
        """ Returns the distinct words from self.text. 
        
            @TODO Should we ever consider case when substituting words?
        """
        return self.raw_words

    def text_window_around_index(self, index, size):
        length = len(self.raw_words)
        half_size = (size - 1) // 2
        if index - half_size < 0:
            start = 0
            end = min(size, length-1)
        elif index + half_size > length - 1:
            start = max(0, length - size)
            end = length - 1
        else:
            start = index - half_size
            end = index + half_size
        text_idx_start = self._text_index_of_word_index(start)
        text_idx_end = self._text_index_of_word_index(end) + len(self.raw_words[end])
        return self.text[text_idx_start:text_idx_end]
         
    def _text_index_of_word_index(self, i):
        pre_words = self.raw_words[:i+1]
        lower_text = self.text.lower()
        # Find all words until `i` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word, look_after_index)
        return look_after_index 

    def text_until_word_index(self, i):
        """ Returns the text before the beginning of word at index `i`. """
        look_after_index = self._text_index_of_word_index(i)
        return self.text[:look_after_index]
    
    def text_after_word_index(self, i):
        """ Returns the text after the end of word at index `i`. """
        # Get index of beginning of word then jump to end of word.
        look_after_index = self._text_index_of_word_index(i) + len(self.raw_words[i])
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

    def replace_words_at_indices(self, indices, words, modified_index=None):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        if len(indices) != len(words):
            raise ValueError(f'Cannot replace {len(words)} words at {len(indices)} indices.')
        new_words = self.raw_words[:]
        for i, word in zip(indices, words):
            new_words[i] = word
        return self.replace_new_words(new_words, modified_index=modified_index)
    
    def replace_word_at_index(self, index, new_word):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        return self.replace_words_at_indices([index], [new_word], index)
    
    def replace_new_words(self, new_words, modified_index=None):
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
        return TokenizedText(self.model, final_sentence, modified_index=modified_index)
        
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
