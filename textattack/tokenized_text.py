from textattack.utils import get_device

class TokenizedText:
    def __init__(self, text, text_to_tokens_converter, tokens_to_ids_converter, 
                attack_attrs=dict()):
        """ Initializer stores text and tensor of tokenized text.
        
        Args:
            text (string): The string that this TokenizedText represents
            text_to_tokens_converter (func): a function that can take a string,
                tokenize it, and return the list of tokens
            text_to_tokens_converter (func): takes a list of tokens and returns
                a list of corresponding IDs
        """
        self.text = text
        self.tokens = text_to_tokens_converter(text)
        self.ids = tokens_to_ids_converter(self.tokens)
        self.text_to_tokens_converter = text_to_tokens_converter
        self.tokens_to_ids_converter = tokens_to_ids_converter
        self.raw_words = raw_words(text)
        self.attack_attrs = attack_attrs
    
    def words(self):
        """ Returns the distinct words from self.text. 
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

    def replace_words_at_indices(self, indices, words):
        """ This code returns a new TokenizedText object where the words at 
            `indices` is replaced with the corresponding word from `words`."""
        if len(indices) != len(words):
            raise ValueError(f'Cannot replace {len(words)} words at {len(indices)} indices.')
        new_words = self.raw_words[:]
        for i, word in zip(indices, words):
            new_words[i] = word
        return self._replace_with_new_words(new_words)
    
    def replace_word_at_index(self, index, new_word):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        self.attack_attrs['modified_word_index'] = index
        return self.replace_words_at_indices([index], [new_word])
    
    def _replace_with_new_words(self, new_words):
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
        return TokenizedText(final_sentence, self.text_to_tokens_converter, 
            self.tokens_to_ids_converter, attack_attrs=self.attack_attrs)
    
    def replace_tokens_at_indices(self, indices, tokens_to_replace):
        """ This code returns a new TokenizedText object where the tokens at 
            `index` is replaced with a new word."""
        if len(indices) != len(tokens_to_replace):
            raise ValueError(f'Cannot replace {len(words)} words at {len(indices)} indices.')
        # import pdb; pdb.set_trace()
        new_tokens = self.tokens[:]
        for i, token in zip(indices, tokens_to_replace):
            new_tokens[i] = token
        return self._replace_with_new_tokens(new_tokens)
    
    def replace_token_at_index(self, index, new_token):
        """ Replaces token at index `index` with `new_token`. """
        self.attack_attrs['modified_token_index'] = index
        print('replacing:', self.tokens[index], 'with:', new_token)
        return self.replace_tokens_at_indices([index], [new_token])
    
    def _replace_with_new_tokens(self, new_tokens):
        """ This code returns a new TokenizedText object and replaces old list 
            of tokens with a new list of tokens, but preserves the punctuation 
            and spacing of the original message.
        """
        final_sentence = ''
        text = self.text
        # print("replacing tokens:", self.tokens)
        # print("with tokens:", new_tokens)
        for input_token, new_token in zip(self.tokens, new_tokens):
            # print('input_token', input_token)
            # print('new_token', new_token)
            if is_invisible_token(input_token) or is_invisible_token(new_token):
                continue
            input_token = clean_token_str(input_token)
            new_token = clean_token_str(new_token)
            # print('\t[2]input_token', input_token)
            # print('\t[2]new_token', new_token)
            token_start = text.lower().index(input_token)
            token_end = token_start + len(input_token)
            final_sentence += text[:token_start]
            final_sentence += new_token
            text = text[token_end:]
        final_sentence += text # Add all of the ending punctuation.
        return TokenizedText(final_sentence, self.text_to_tokens_converter, 
            self.tokens_to_ids_converter, attack_attrs=self.attack_attrs)
        
    def __repr__(self):
        return f'<TokenizedText "{self.text}">'

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

def is_invisible_token(token_str):
    """ Returns true if this token does not appear in the actual string,
        like '<oov>' or '[CLS]'. """
    return token_str in ['<oov>', '[PAD]', '[CLS]', '[SEP]']

def clean_token_str(token_str):
    """ Converts a sub-word token to how it would actually appear in a string. """
    return token_str.replace('##', '')