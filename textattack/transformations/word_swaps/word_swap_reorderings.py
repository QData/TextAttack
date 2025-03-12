import textattack

from .word_swap import WordSwap

from typing import List
from textattack.shared import AttackedText


class WordSwapReorderings(WordSwap):
    """Generates visually identical reorderings of a string using swap and encoding procedures."""


    def __init__(self):
        super().__init__()
        self.PDF = chr(0x202C)
        self.LRE = chr(0x202A)
        self.RLE = chr(0x202B)
        self.LRO = chr(0x202D)
        self.RLO = chr(0x202E)
        self.PDI = chr(0x2069)
        self.LRI = chr(0x2066)
        self.RLI = chr(0x2067)

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

    # def some(self, *els):
    #     """Returns the arguments as a tuple with Nones removed."""
    #     return tuple(filter(None, els))

    # def swaps(self, chars: str) -> set:
    #     """Generates all possible swaps for a string."""
    #     def pairs(chars, pre=(), suf=()):
    #         orders = set()
    #         for i in range(len(chars)-1):
    #             prefix = pre + tuple(chars[:i])
    #             suffix = suf + tuple(chars[i+2:])
    #             swap = self.Swap(chars[i+1], chars[i])
    #             pair = self.some(prefix, swap, suffix)
    #             orders.add(pair)
    #             orders.update(pairs(suffix, pre=some(prefix, swap)))
    #             orders.update(pairs(self.some(prefix, swap), suf=suffix))
    #         return orders
    #     return pairs(chars) | {tuple(chars)}

    # def unswap(self, el: tuple) -> str:
    #     """Reverts a tuple of swaps to the original string."""
    #     if isinstance(el, str):
    #         return el
    #     elif isinstance(el, self.Swap):
    #         return self.unswap((el.two, el.one))
    #     else:
    #         res = ""
    #         for e in el:
    #             res += self.unswap(e)
    #         return res

    # def uniswap(self, els):
    #     """Encodes the elements into a Unicode Bidi representation."""
    #     res = ""
    #     for el in els:
    #         if isinstance(el, self.Swap):
    #             res += self.uniswap([
    #                 self.LRO, self.LRI, self.RLO, self.LRI,
    #                 el.one, self.PDI, self.LRI, el.two, self.PDI,
    #                 self.PDF, self.PDI, self.PDF
    #             ])
    #         elif isinstance(el, str):
    #             res += el
    #         else:
    #             for subel in el:
    #                 res += self.uniswap([subel])
    #     return res

    def _get_replacement_words(self, word):
        candidate_words = []
        return candidate_words

    # def strings_to_file(self, file, string):
    #     """Writes all reordered strings to a file."""
    #     with open(file, 'w') as f:
    #         for swap in self.swaps(string):
    #             uni = self.uniswap(swap)
    #             print(uni, file=f)

    # def print_strings(self, string):
    #     """Prints all reordered strings."""
    #     for swap in self.swaps(string):
    #         uni = self.uniswap(swap)
    #         print(uni)

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def bounds(self, sentence, max_perturbs):
        return [(-1, len(sentence.text) - 1)] * max_perturbs

    def apply_perturbation(self, sentence, perturbation_vector: List[float]): # AttackedText object to AttackedText object
        def swaps(els) -> str:
            res = ""
            for el in els:
                if isinstance(el, self.Swap):
                    res += swaps([self.LRO, self.LRI, self.RLO, self.LRI, el.one, self.PDI, self.LRI, el.two, self.PDI, self.PDF, self.PDI, self.PDF])
                elif isinstance(el, str):
                    res += el
                else:
                    for subel in el:
                        res += swaps([subel])
            return res
        candidate = list(sentence.text)
        for perturb in map(self.natural, perturbation_vector):
            if (perturb >= 0 and len(candidate) >= 2):
                perturb = min(perturb, len(candidate) - 2)
                candidate = candidate[:perturb] + [self.Swap(candidate[perturb+1], candidate[perturb])] + candidate[perturb+2:]

        return AttackedText(swaps(candidate))