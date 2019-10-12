from utils import get_device

class TokenizedText:
    def __init__(self, model, text):
        """ Initializer stores text and tensor of tokenized text.
        """
        self.model = model
        self.text = text
        self.ids = model.convert_text_to_ids(text)
        print('TokenizedText:', self.text, '/', self.words())
    
    def words(self):
        """ Returns the distinct words from self.text. """
        # @TODO be smarter here. This won't work. Do simple tokenization and 
        # convert words to/from lowercase.
        return self.text.split(' ')
    
    def first_word_diff(self, other_words):
        """ Returns the index of the first word in self.words() that differs
            from other_words. Useful for word swap strategies. """
        sw = self.words()
        for i in min(len(sw), len(other_words)):
            if sw[i] != other_words[i]:
                return i
        return None
    
    def replace_word_at_index(self, index, new_word):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        print('self.words(): ', self.words(),
            '/', self.text)
        print('index/new_words:', index, new_word)
        new_words = self.words()[:]
        new_words[index] = new_word
        return self.replace_new_words(new_words)
    
    def replace_new_words(self, adv_words):
        """ This code returns a new TokenizedText object and replaces old list 
            of words with a new list of words, but preserves the punctuation 
            and spacing of the original message.
        """
        final_sentence = ''
        text = self.text
        for input_word, adv_word in zip(self.words(), adv_words):
            if input_word == '[UNKNOWN]': continue
            word_start = text.index(input_word)
            word_end = word_start + len(input_word)
            final_sentence += text[:word_start]
            final_sentence += adv_word
            text = text[word_end:]
        final_sentence += text # Add all of the ending punctuation.
        print('replace_new_words:', adv_words)
        print('final_sentence:', final_sentence)
        return TokenizedText(self.model, final_sentence)
        