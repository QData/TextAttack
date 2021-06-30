"""
Shared TextAttack Functions
=============================

This package includes functions shared across packages.

"""


from . import data
from . import utils
from .utils import logger
from . import validators

from .attacked_text import AttackedText
from .word_embeddings import AbstractWordEmbedding, WordEmbedding, GensimWordEmbedding
from .checkpoint import AttackCheckpoint
