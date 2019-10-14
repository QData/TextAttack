from utils import get_device

class TokenizedText:
    def __init__(self, model, text):
        """ Initializer stores text and tensor of tokenized text.
        """
        self.model = model
        self.text = text
        self.ids = model.convert_text_to_ids(text)
    
    def words(self):
        """ Returns the distinct words from self.text. 
        
            @TODO Should we consider case when substituting words?
        """
        return raw_words(self.text)
    
    def first_word_diff(self, other_tokenized_text):
        """ Returns the index of the first word in self.words() that differs
            from other_words. Useful for word swap strategies. """
        w1 = self.words()
        w2 = other_tokenized_text.words()
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return i
        return None
    
    def replace_words_at_indices(self, indices, words):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        if len(indices) != len(words):
            raise ValueError(f'Cannot replace {len(words)} words at {len(indices)} indices.')
        new_words = self.words()[:]
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
        for input_word, adv_word in zip(self.words(), new_words):
            if input_word == '[UNKNOWN]': continue
            word_start = text.index(input_word)
            word_end = word_start + len(input_word)
            final_sentence += text[:word_start]
            final_sentence += adv_word
            text = text[word_end:]
        final_sentence += text # Add all of the ending punctuation.
        return TokenizedText(self.model, final_sentence)
    
    def __str__(self):
        return self.text

def raw_words(s):
    """ Lowercases a string, removes all non-alphanumeric characters,
        and splits into words. """
    words = []
    word = ''
    for x in ' '.join(s.split()):
        c = x.lower()
        if c in 'abcdefghijklmnopqrstuvwxyz':
            word += c
        elif word:
            words.append(word)
            word = ''
    if word: words.append(word)
    return words
