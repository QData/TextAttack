import textattack

from .word_swap import WordSwap


class WordSwapReorderings(WordSwap):
    """Generates visually identical reorderings of a string using swap and encoding procedures."""


    def __init__(self):
        super().__init__()
        self.PDF = "\u202C"
        self.LRO = "\u202D"
        self.RLO = "\u202E"
        self.PDI = "\u2069"
        self.LRI = "\u2066"

    class Swap:
        """Represents swapped elements in a string of text."""
        def __init__(self, one, two):
            self.one = one
            self.two = two

        def __repr__(self):
            return f"Swap({self.one}, {self.two})"

        def __eq__(self, other):
            return self.one == other.one and self.two == other.two

        def __hash__(self):
            return hash((self.one, self.two))

    @staticmethod
    def _some(*els):
        """Filters out None values and returns a tuple."""
        return tuple(filter(None, els))

    def _generate_swaps(self, chars):
        """Generates all possible swaps for a string."""
        def pairs(chars, pre=(), suf=()):
            orders = set()
            for i in range(len(chars) - 1):
                prefix = pre + tuple(chars[:i])
                suffix = suf + tuple(chars[i + 2:])
                swap = self.Swap(chars[i + 1], chars[i])
                pair = self._some(prefix, swap, suffix)
                orders.add(pair)
                # Recursive calls
                orders.update(pairs(suffix, pre=self._some(prefix, swap)))
                orders.update(pairs(self._some(prefix, swap), suf=suffix))
            return orders

        return pairs(chars) | {tuple(chars)}

    def _unswap(self, el):
        """Reverts a tuple of swaps to the original string."""
        if isinstance(el, str):
            return el
        elif isinstance(el, self.Swap):
            return self._unswap((el.two, el.one))
        else:
            res = ""
            for e in el:
                res += self._unswap(e)
            return res

    def _uniswap(self, els):
        """Encodes the elements into a Unicode Bidi representation."""
        res = ""
        for el in els:
            if isinstance(el, self.Swap):
                res += self._uniswap([
                    self.LRO, self.LRI, self.RLO, self.LRI,
                    el.one, self.PDI, self.LRI, el.two, self.PDI,
                    self.PDF, self.PDI, self.PDF
                ])
            elif isinstance(el, str):
                res += el
            else:
                for subel in el:
                    res += self._uniswap([subel])
        return res

    def _get_replacement_words(self, word):
        """
        Returns a list of visually identical reorderings of the input word
        encoded with Unicode Bidi characters.
        """
        # Edge case: Single character words cannot be reordered
        if len(word) <= 1:
            return []

        # Generate all possible swaps of the word
        orderings = self._generate_swaps(word)

        # Encode each ordering into a visually identical Unicode string
        replacements = [self._uniswap(ordering) for ordering in orderings]

        return replacements


    def strings_to_file(self, file, string):
        """Writes all reordered strings to a file."""
        with open(file, 'w') as f:
            for swap in self._generate_swaps(string):
                uni = self._uniswap(swap)
                print(uni, file=f)

    def display_control_characters(self, s):
        """
        Replaces Unicode control characters with visible placeholders for debugging.
        """
        control_map = {
            "\u202C": "<PDF>",
            "\u202D": "<LRO>",
            "\u202E": "<RLO>",
            "\u2069": "<PDI>",
            "\u2066": "<LRI>",
        }
        return "".join(control_map.get(c, c) for c in s)

    def print_strings(self, string):
        """Prints all reordered strings."""
        for swap in self._generate_swaps(string):
            uni = self._uniswap(swap)
            # print(self.display_control_characters(uni))
            print(repr(uni))
