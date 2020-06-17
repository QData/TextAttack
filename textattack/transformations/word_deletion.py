from textattack.transformations.word_swap import WordSwap

from .transformation import Transformation


class WordDeletion(Transformation):
    """
    An abstract class that takes a sentence and transforms it by replacing
    deleting a single word.
    
        letters_to_insert (string): letters allowed for insertion into words
    """

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []
        for i in indices_to_modify:
            transformed_texts.append(current_text.delete_word_at_index(i))
        return transformed_texts
