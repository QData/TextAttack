"""
"Learning To Write" Language Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import torch

import textattack
from textattack.constraints.grammaticality.language_models import (
    LanguageModelConstraint,
)

from .language_model_helpers import QueryHandler


class LearningToWriteLanguageModel(LanguageModelConstraint):
    """A constraint based on the L2W language model.

    The RNN-based language model from "Learning to Write With
    Cooperative Discriminators" (Holtzman et al, 2018).

    https://arxiv.org/pdf/1805.06087.pdf

    https://github.com/windweller/l2w

     Reused by Jia et al., 2019, as a substitution for the Google
    1-billion words language model (in a revised version the attack of
    Alzantot et al., 2018).

    https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689
    """

    CACHE_PATH = "constraints/grammaticality/language-models/learning-to-write"

    def __init__(self, window_size=5, **kwargs):
        self.window_size = window_size
        lm_folder_path = textattack.shared.utils.download_from_s3(
            LearningToWriteLanguageModel.CACHE_PATH
        )
        self.query_handler = QueryHandler.load_model(
            lm_folder_path, textattack.shared.utils.device
        )
        super().__init__(**kwargs)

    def get_log_probs_at_index(self, text_list, word_index):
        """Gets the probability of the word at index `word_index` according to
        the language model."""
        queries = []
        query_words = []
        for attacked_text in text_list:
            word = attacked_text.words[word_index]
            window_text = attacked_text.text_window_around_index(
                word_index, self.window_size
            )
            query = textattack.shared.utils.words_from_text(window_text)
            queries.append(query)
            query_words.append(word)
        log_probs = self.query_handler.query(queries, query_words)
        return torch.tensor(log_probs)
