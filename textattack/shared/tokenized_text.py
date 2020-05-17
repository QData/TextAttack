import torch
from copy import deepcopy
from .utils import get_device, words_from_text
from nltk.corpus import stopwords

class TokenizedText:

    """ 
     A helper class that represents a string that can be attacked.
     
     Models that take multiple sentences as input separate them by `SPLIT_TOKEN`. 
     Attacks "see" the entire input, joined into one string, without the split token. 

     Args:
        text (string): The string that this TokenizedText represents
        tokenizer (textattack.Tokenizer): An object that can encode text
        
    """
   
    SPLIT_TOKEN = '>>>>'
    
    def __init__(self, text, tokenizer, attack_attrs=dict()):   
        text = text.strip()
        self.tokenizer = tokenizer
        ids = tokenizer.encode(text)
        if not isinstance(ids, tuple):
            # Some tokenizers may tokenize text to a single vector.
            # In this case, wrap the vector in a tuple to mirror the 
            # format of other tokenizers.
            ids = (ids,)
        self.ids = ids
        self.words = words_from_text(text, words_to_ignore=[TokenizedText.SPLIT_TOKEN])
        self.text = text
        self.attack_attrs = attack_attrs
        self._identify_stopwords()
        if 'modified_indices' not in attack_attrs:
            attack_attrs['modified_indices'] = set()

    def __eq__(self, other):
        return (self.text == other.text) and (self.attack_attrs == other.attack_attrs)
    
    def __hash__(self):
        return hash(self.text)

    def _identify_stopwords(self):
        self.stopwords = set(stopwords.words('english'))
        if 'textfooler_stopwords' in attack_attrs:
            self.stopwords = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
        
        attack_attrs['stopword_indices'] = set()
        for i, word in enumerate(self.words):
            if word.lower() in self.stopwords:
                attack_attrs['stopword_indices'].add(i)

    def delete_tensors(self):
        """ Delete tensors to clear up GPU space. Only should be called
            once the TokenizedText is only needed to display.
        """
        self.ids = None
        for key in self.attack_attrs:
            if isinstance(self.attack_attrs[key], torch.Tensor):
                del self.attack_attrs[key]

    def text_window_around_index(self, index, window_size):
        """ The text window of `window_size` words centered around `index`. """
        length = len(self.words)
        half_size = (window_size - 1) // 2
        if index - half_size < 0:
            start = 0
            end = min(window_size, length-1)
        elif index + half_size > length - 1:
            start = max(0, length - window_size)
            end = length - 1
        else:
            start = index - half_size
            end = index + half_size
        text_idx_start = self._text_index_of_word_index(start)
        text_idx_end = self._text_index_of_word_index(end) + len(self.words[end])
        return self.text[text_idx_start:text_idx_end]
         
    def _text_index_of_word_index(self, i):
        """ Returns the index of word `i` in self.text. """
        pre_words = self.words[:i+1]
        lower_text = self.text.lower()
        # Find all words until `i` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word.lower(), look_after_index)
        return look_after_index 

    def text_until_word_index(self, i):
        """ Returns the text before the beginning of word at index `i`. """
        look_after_index = self._text_index_of_word_index(i)
        return self.text[:look_after_index]
    
    def text_after_word_index(self, i):
        """ Returns the text after the end of word at index `i`. """
        # Get index of beginning of word then jump to end of word.
        look_after_index = self._text_index_of_word_index(i) + len(self.words[i])
        return self.text[look_after_index:]
    
    def first_word_diff(self, other_tokenized_text):
        """ Returns the first word in self.words that differs from 
            other_tokenized_text. Useful for word swap strategies. """
        w1 = self.words
        w2 = other_tokenized_text.words
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return w1
        return None
    
    def first_word_diff_index(self, other_tokenized_text):
        """ Returns the index of the first word in self.words that differs
            from other_tokenized_text. Useful for word swap strategies. """
        w1 = self.words
        w2 = other_tokenized_text.words
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                return i
        return None
   
    def all_words_diff(self, other_tokenized_text):
        """ Returns the set of indices for which this and other_tokenized_text
        have different words. """
        indices = set()
        w1 = self.words
        w2 = other_tokenized_text.words
        for i in range(min(len(w1), len(w2))):
            if w1[i] != w2[i]:
                indices.add(i)
        return indices
        
    def ith_word_diff(self, other_tokenized_text, i):
        """ Returns whether the word at index i differs from other_tokenized_text
        """
        w1 = self.words
        w2 = other_tokenized_text.words
        if len(w1) - 1 < i or len(w2) - 1 < i:
            return True
        return w1[i] != w2[i]

    def replace_words_at_indices(self, indices, new_words):
        """ This code returns a new TokenizedText object where the word at 
            `index` is replaced with a new word."""
        if len(indices) != len(new_words):
            raise ValueError(f'Cannot replace {len(new_words)} words at {len(indices)} indices.')
        words = self.words[:]
        for i, new_word in zip(indices, new_words):
            words[i] = new_word
        return self.replace_new_words(words)
    
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
        new_attack_attrs = deepcopy(self.attack_attrs)
        new_attack_attrs['stopword_indices'] = set()
        new_attack_attrs['modified_indices'] = set()
        new_attack_attrs['newly_modified_indices'] = set()
        new_i = 0
        for i, (input_word, adv_word) in enumerate(zip(self.words, new_words)):
            if input_word == '[DELETE]': continue
            word_start = text.index(input_word)
            word_end = word_start + len(input_word)
            final_sentence += text[:word_start]
            final_sentence += adv_word
            text = text[word_end:]
            if i in self.attack_attrs['stopword_indices']:
                new_attack_attrs['stopword_indices'].add(new_i)
            if i in self.attack_attrs['modified_indices'] or input_word != adv_word:
                new_attack_attrs['modified_indices'].add(new_i)
                if input_word != adv_word:
                    new_attack_attrs['newly_modified_indices'].add(new_i)
            new_i += 1
        final_sentence += text # Add all of the ending punctuation.
        return TokenizedText(final_sentence, self.tokenizer, 
            attack_attrs=new_attack_attrs)
    
    def clean_text(self):
        """ Represents self in a clean, printable format. Joins text with multiple
            inputs separated by `TokenizedText.SPLIT_TOKEN` with a line break.
        """
        return self.text.replace(TokenizedText.SPLIT_TOKEN, '\n\n')
    
    def __repr__(self):
        return f'<TokenizedText "{self.text}">'
