import random
from .word_swap import WordSwap

class WordSwapDeletions(WordSwap):
    """
    Generates text transformations by embedding Unicode control characters 
    (e.g., backspace, delete, carriage return) into strings to manipulate
    how they render or interact with models.
    """

    def __init__(self):
        super().__init__()
        self.BKSP = chr(0x8)  # Backspace character
        self.DEL = chr(0x7F)  # Delete character
        self.CR = chr(0xD)    # Carriage return character

    def _generate_transformations(self, text):
        """
        Inserts deletion-related control characters into text to create transformations.

        Args:
            text (str): The input string.

        Returns:
            List[str]: A list of transformed strings with deletion characters.
        """
        transformations = []

        # Insert BKSP (Backspace) characters
        for i in range(len(text) + 1):
            transformations.append(text[:i] + self.BKSP * max(1, len(text) - i))

        # Insert DEL (Delete) characters
        for i in range(len(text) + 1):
            transformations.append(text[:i] + self.DEL + text[i:])

        # Insert CR (Carriage Return) characters
        for i in range(1, len(text) + 1):
            transformations.append(text[:i] + self.CR + text[i:])

        return transformations

    def _get_replacement_words(self, word):
        """
        Returns transformations for a given word by embedding deletion-related control characters.

        Args:
            word (str): The input word.

        Returns:
            List[str]: A list of transformed strings with deletion characters.
        """
        if len(word) <= 1:
            return []  # No meaningful transformations for single-character words

        return self._generate_transformations(word)

    def print_transformations(self, word):
        """
        Prints all transformations of the input word for debugging or demonstration purposes.

        Args:
            word (str): The input word.
        """
        transformations = self._get_replacement_words(word)
        for t in transformations:
            print(t)
