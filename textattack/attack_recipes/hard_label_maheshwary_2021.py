"""
Hard Label Black Box Attack
==================================
(Generating Natural Language Attack in a Hard Label Black Box Setting)
"""
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import HardLabelAttack
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class HardLabelMaheshwary2021(AttackRecipe):
    """Rishabh Maheshwary, Saket Maheshwary, Vikram Pudi (2021).

    Generating Natural Language Attack in a Hard Label Black Box
    Setting. https://arxiv.org/abs/2012.14956 Methodology description
    quoted from the paper: We propose a decision-based attack strategy
    that crafts high quality adversarial examples on text classification
    and entailment tasks. Our proposed attack strategy leverages
    population-based optimization algorithm to craft plausible and
    semantically similar adversarial examples by observing only the top
    label predicted by the target model. At each iteration, the
    optimization procedure allow word replacements that maximizes the
    overall semantic similarity between the original and the adversarial
    text.
    """

    @staticmethod
    def build(model):
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the stopwords defined
        # in the paper public implementation.
        #
        stopwords = set(
            [
                "a",
                "about",
                "above",
                "across",
                "after",
                "afterwards",
                "again",
                "against",
                "ain",
                "all",
                "almost",
                "alone",
                "along",
                "already",
                "also",
                "although",
                "am",
                "among",
                "amongst",
                "an",
                "and",
                "another",
                "any",
                "anyhow",
                "anyone",
                "anything",
                "anyway",
                "anywhere",
                "are",
                "aren",
                "aren't",
                "around",
                "as",
                "at",
                "back",
                "been",
                "before",
                "beforehand",
                "behind",
                "being",
                "below",
                "beside",
                "besides",
                "between",
                "beyond",
                "both",
                "but",
                "by",
                "can",
                "cannot",
                "could",
                "couldn",
                "couldn't",
                "d",
                "didn",
                "didn't",
                "doesn",
                "doesn't",
                "don",
                "don't",
                "down",
                "due",
                "during",
                "either",
                "else",
                "elsewhere",
                "empty",
                "enough",
                "even",
                "ever",
                "everyone",
                "everything",
                "everywhere",
                "except",
                "first",
                "for",
                "former",
                "formerly",
                "from",
                "hadn",
                "hadn't",
                "hasn",
                "hasn't",
                "haven",
                "haven't",
                "he",
                "hence",
                "her",
                "here",
                "hereafter",
                "hereby",
                "herein",
                "hereupon",
                "hers",
                "herself",
                "him",
                "himself",
                "his",
                "how",
                "however",
                "hundred",
                "i",
                "if",
                "in",
                "indeed",
                "into",
                "is",
                "isn",
                "isn't",
                "it",
                "it's",
                "its",
                "itself",
                "just",
                "latter",
                "latterly",
                "least",
                "ll",
                "may",
                "me",
                "meanwhile",
                "mightn",
                "mightn't",
                "mine",
                "more",
                "moreover",
                "most",
                "mostly",
                "must",
                "mustn",
                "mustn't",
                "my",
                "myself",
                "namely",
                "needn",
                "needn't",
                "neither",
                "never",
                "nevertheless",
                "next",
                "no",
                "nobody",
                "none",
                "noone",
                "nor",
                "not",
                "nothing",
                "now",
                "nowhere",
                "o",
                "of",
                "off",
                "on",
                "once",
                "one",
                "only",
                "onto",
                "or",
                "other",
                "others",
                "otherwise",
                "our",
                "ours",
                "ourselves",
                "out",
                "over",
                "per",
                "please",
                "s",
                "same",
                "shan",
                "shan't",
                "she",
                "she's",
                "should've",
                "shouldn",
                "shouldn't",
                "somehow",
                "something",
                "sometime",
                "somewhere",
                "such",
                "t",
                "than",
                "that",
                "that'll",
                "the",
                "their",
                "theirs",
                "them",
                "themselves",
                "then",
                "thence",
                "there",
                "thereafter",
                "thereby",
                "therefore",
                "therein",
                "thereupon",
                "these",
                "they",
                "this",
                "those",
                "through",
                "throughout",
                "thru",
                "thus",
                "to",
                "too",
                "toward",
                "towards",
                "under",
                "unless",
                "until",
                "up",
                "upon",
                "used",
                "ve",
                "was",
                "wasn",
                "wasn't",
                "we",
                "were",
                "weren",
                "weren't",
                "what",
                "whatever",
                "when",
                "whence",
                "whenever",
                "where",
                "whereafter",
                "whereas",
                "whereby",
                "wherein",
                "whereupon",
                "wherever",
                "whether",
                "which",
                "while",
                "whither",
                "who",
                "whoever",
                "whole",
                "whom",
                "whose",
                "why",
                "with",
                "within",
                "without",
                "won",
                "won't",
                "would",
                "wouldn",
                "wouldn't",
                "y",
                "yet",
                "you",
                "you'd",
                "you'll",
                "you're",
                "you've",
                "your",
                "yours",
                "yourself",
                "yourselves",
            ]
        )
        # fmt: on
        constraints = [StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Minimum word embedding cosine similarity of 0.5.
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Generate attack using only the topmost predicted label.
        #
        search_method = HardLabelAttack(pop_size=30, max_iters=100)

        return Attack(goal_function, constraints, transformation, search_method)
