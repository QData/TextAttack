import torch
from .utils import get_device, words_from_text

class TokenizedText:

    """ 
     A helper class that represents a string that can be attacked.
     
     Models that take multiple sentences as input separate them by ``SPLIT_TOKEN``. 
     Attacks "see" the entire input, joined into one string, without the split token. 

     Args:
        text (string): The string that this TokenizedText represents
        tokenizer (`TextAttack.Tokenizer`): An object that can encode text
        
    """
   
    SPLIT_TOKEN = '>>>>'
    
    def __init__(self, text, tokenizer, attack_attrs=dict()):   
        text = text.strip()
        self.tokenizer = tokenizer
        if tokenizer:
            ids = tokenizer.encode(text)
            if not isinstance(ids, tuple):
                # Some tokenizers may tokenize text to a single vector.
                # In this case, wrap the vector in a tuple to mirror the 
                # format of other tokenizers.
                ids = (ids,)
            self.ids = ids
        else:
            self.ids = None
        self.words = words_from_text(text, words_to_ignore=[TokenizedText.SPLIT_TOKEN])
        self.text = text
        self.attack_attrs = attack_attrs
        # Indices of deleted words from the *original* text.
        self.attack_attrs.setdefault('deletion_indices', set())
        # Indices of inserted words in the *perturbed* text (this).
        self.attack_attrs.setdefault('insertion_indices', set())
        self.attack_attrs.setdefault('modified_indices', set()) # @TODO replace ``modified`` with ``modification``

    def __eq__(self, other):
        return (self.text == other.text) and (self.attack_attrs == other.attack_attrs)
    
    def __hash__(self):
        return hash(self.text)

    def free_memory(self):
        """ Delete items that take up memory.
            
            Can be called once the TokenizedText is only needed to display.
        """
        self.ids = None
        self.tokenizer = None
        if 'last_transformation' in self.attack_attrs:
            del self.attack_attrs['last_transformation']
        for key in self.attack_attrs:
            if isinstance(self.attack_attrs[key], torch.Tensor):
                del self.attack_attrs[key]

    def text_window_around_index(self, index, window_size):
        """ The text window of ``window_size`` words centered around ``index``. """
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
        """ Returns the index of word ``i`` in self.text. """
        pre_words = self.words[:i+1]
        lower_text = self.text.lower()
        # Find all words until `i` in string.
        look_after_index = 0
        for word in pre_words:
            look_after_index = lower_text.find(word.lower(), look_after_index)
        return look_after_index 

    def text_until_word_index(self, i):
        """ Returns the text before the beginning of word at index ``i``. """
        look_after_index = self._text_index_of_word_index(i)
        return self.text[:look_after_index]
    
    def text_after_word_index(self, i):
        """ Returns the text after the end of word at index ``i``. """
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
            ``index`` is replaced with a new word."""
        if len(indices) != len(new_words):
            raise ValueError(f'Cannot replace {len(new_words)} words at {len(indices)} indices.')
        words = self.words[:]
        for i, new_word in zip(indices, new_words):
            words[i] = new_word
        return self.replace_new_words(words)
    
    def replace_word_at_index(self, index, new_word):
        """ This code returns a new TokenizedText object where the word at 
            ``index`` is replaced with a new word."""
        return self.replace_words_at_indices([index], [new_word])
    
    def replace_new_words(self, new_words):
        """ This code returns a new TokenizedText object and replaces old list 
            of words with a new list of words, but preserves the punctuation 
            and spacing of the original message.
        """
        final_sentence = ''
        text = self.text
        new_attack_attrs = dict()
        new_attack_attrs['deletion_indices'] = self.attack_attrs['deletion_indices'].copy()
        new_attack_attrs['insertion_indices'] = self.attack_attrs['insertion_indices'].copy()
        new_attack_attrs['modified_indices'] = set()
        new_attack_attrs['newly_modified_indices'] = set()
        new_i = 0
        for i, (input_word, adv_word) in enumerate(zip(self.words, new_words)):
            if input_word == '[DELETE]': continue
            word_start = text.index(input_word)
            word_end = word_start + len(input_word)
            final_sentence += text[:word_start]
            text = text[word_end:]
            adv_num_words = len(words_from_text(adv_word))
            if adv_num_words == 0:
                # Re-calculated modified indices.
                modified_indices = list(new_attack_attrs['newly_modified_indices'])
                new_modified_indices = set()
                for j, modified_idx in enumerate(modified_indices):
                    if modified_idx < i:
                        new_modified_indices.add(modified_idx)
                    elif modified_idx > i:
                        new_modified_indices.add(modified_idx-1)
                    else:
                        pass
                        # @TODO what if modified_idx == i? is it correct to just remove?
                new_attack_attrs['new_modified_indices'] = new_modified_indices
                # Re-calculated deleted index.
                deleted_idx = i
                for other_deleted_idx in sorted(new_attack_attrs['deletion_indices']):
                    if other_deleted_idx < deleted_idx:
                        deleted_idx += 1
                # Track deleted words.
                new_attack_attrs['deletion_indices'].add(deleted_idx)
                print('idx', deleted_idx, 'deleting i',i,'input_word',input_word,'adv_word',adv_word, new_attack_attrs['deletion_indices'])
                # if i > 18: import pdb; pdb.set_trace()
                # Remove extra space (or else there would be two spaces for each
                # deleted word).
                # @TODO What to do with punctuation in this case? This behavior is undefined
                if i == 0:
                    # If the first word was deleted, take a subsequent space.
                    if text[0] == ' ':
                        text = text[1:]
                else:
                    # If a word other than the first was deleted, take a preceding space.
                    if final_sentence[-1] == ' ':
                        final_sentence = final_sentence[:-1]
            elif adv_num_words == 1:
                if i in self.attack_attrs['modified_indices'] or input_word != adv_word:
                    new_attack_attrs['modified_indices'].add(new_i)
                    if input_word != adv_word:
                        new_attack_attrs['newly_modified_indices'].add(new_i)
            elif adv_num_words > 0:
                import textattack
                textattack.shared.utils.get_logger().warn(f'havent yet implemented multiword subs - cant handle {adv_word} / {adv_num_words} / {words_from_text(adv_word)}')
                if i in self.attack_attrs['modified_indices'] or input_word != adv_word:
                    new_attack_attrs['modified_indices'].add(new_i)
                    if input_word != adv_word:
                        new_attack_attrs['newly_modified_indices'].add(new_i)
            else:
                raise ValueError(f'perturbed sequence has {adv_num_words} words')
            final_sentence += adv_word
            new_i += 1
        final_sentence += text # Add all of the ending punctuation.
        return TokenizedText(final_sentence, self.tokenizer, 
            attack_attrs=new_attack_attrs)
    
    def clean_text(self):
        """ Represents self in a clean, printable format. Joins text with multiple
            inputs separated by ``TokenizedText.SPLIT_TOKEN`` with a line break.
        """
        return self.text.replace(TokenizedText.SPLIT_TOKEN, '\n\n')
    
    def __repr__(self):
        return f'<TokenizedText "{self.text}">'
