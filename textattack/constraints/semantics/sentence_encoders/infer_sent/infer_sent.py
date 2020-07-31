import os

import torch

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared import utils

from .infer_sent_model import InferSentModel


class InferSent(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using InferSent."""

    MODEL_PATH = "constraints/semantics/sentence-encoders/infersent-encoder"
    WORD_EMBEDDING_PATH = "word_embeddings"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.get_infersent_model()
        self.model.to(utils.device)

    def get_infersent_model(self):
        """Retrieves the InferSent model.

        Returns:
            The pretrained InferSent model.
        """
        infersent_version = 2
        model_folder_path = utils.download_if_needed(InferSent.MODEL_PATH)
        model_path = os.path.join(
            model_folder_path, f"infersent{infersent_version}.pkl"
        )
        params_model = {
            "bsize": 64,
            "word_emb_dim": 300,
            "enc_lstm_dim": 2048,
            "pool_type": "max",
            "dpout_model": 0.0,
            "version": infersent_version,
        }
        infersent = InferSentModel(params_model)
        infersent.load_state_dict(torch.load(model_path))
        word_embedding_path = utils.download_if_needed(InferSent.WORD_EMBEDDING_PATH)
        w2v_path = os.path.join(word_embedding_path, "fastText", "crawl-300d-2M.vec")
        infersent.set_w2v_path(w2v_path)
        infersent.build_vocab_k_words(K=100000)
        return infersent

    def encode(self, sentences):
        return self.model.encode(sentences, tokenize=True)
