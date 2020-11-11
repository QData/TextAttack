"""

Google Language Models from Alzantot
--------------------------------------

    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""


import os

import lru
import numpy as np

from textattack.shared import utils

from . import lm_data_utils, lm_utils

tf = utils.LazyLoader("tensorflow", globals(), "tensorflow")


# @TODO automatically choose between GPU and CPU.


class GoogLMHelper:
    """An implementation of `<https://arxiv.org/abs/1804.07998>`_ adapted from
    `<https://github.com/nesl/nlp_adversarial_examples>`_."""

    CACHE_PATH = "constraints/semantics/language-models/alzantot-goog-lm"

    def __init__(self):
        tf.get_logger().setLevel("INFO")
        lm_folder = utils.download_if_needed(GoogLMHelper.CACHE_PATH)
        self.PBTXT_PATH = os.path.join(lm_folder, "graph-2016-09-10-gpu.pbtxt")
        self.CKPT_PATH = os.path.join(lm_folder, "ckpt-*")
        self.VOCAB_PATH = os.path.join(lm_folder, "vocab-2016-09-10.txt")

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = lm_data_utils.CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        with tf.device("/gpu:1"):
            self.graph = tf.Graph()
            self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            self.t = lm_utils.LoadModel(
                self.sess, self.graph, self.PBTXT_PATH, self.CKPT_PATH
            )

        self.lm_cache = lru.LRU(2 ** 18)

    def clear_cache(self):
        self.lm_cache.clear()

    def get_words_probs_uncached(self, prefix_words, list_words):
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find("<S>") != 0:
            prefix_words = "<S> " + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros(
            [self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32
        )

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [[samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(
            self.t["softmax_out"],
            feed_dict={
                self.t["char_inputs_in"]: char_ids_inputs,
                self.t["inputs_in"]: inputs,
                self.t["targets_in"]: targets,
                self.t["target_weights_in"]: weights,
            },
        )
        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs = [softmax[0][w_id] for w_id in words_ids]
        return np.array(word_probs)

    def get_words_probs(self, prefix, list_words):
        """Retrieves the probability of words.

        Args:
            prefix_words
            list_words
        """
        uncached_words = []
        for word in list_words:
            if (prefix, word) not in self.lm_cache:
                if word not in uncached_words:
                    uncached_words.append(word)
        probs = self.get_words_probs_uncached(prefix, uncached_words)
        for word, prob in zip(uncached_words, probs):
            self.lm_cache[prefix, word] = prob
        return [self.lm_cache[prefix, word] for word in list_words]
