"""

WordInsertionRandomSynonym Class
------------------------------------
random synonym insertation Transformation
"""

import random

import string

from .word_insertion import WordInsertion


class WordInsertionRandomHyperlink(WordInsertion):
    """Transformation that inserts synonyms of words that are already in the
    sequence."""

    def generate_random_url(self, length=10):
        """Generate a random URL with a given length."""
        chars = string.ascii_lowercase + string.digits
        return 'http://www.' + ''.join(random.choice(chars) for _ in range(length)) + '.com'

    def generate_random_hyperlink(self):
        """Generate an HTML hyperlink with a random URL."""
        url = self.generate_random_url()
        return f'<a href="{url}">{url}</a>'
    
    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        for idx in indices_to_modify:
            transformed_texts.append(
                current_text.insert_text_before_word_index(idx, self.generate_random_hyperlink())
            )
        return transformed_texts

    @property
    def deterministic(self):
        return False

    def extra_repr_keys(self):
        return super().extra_repr_keys()