"""
SentenceTransformation class
-----------------------------------

https://github.com/makcedward/nlpaug

"""


from eukaryote.transformations import Transformation


class SentenceTransformation(Transformation):
    def _get_transformations(self, current_text, indices_to_modify):
        raise NotImplementedError()
