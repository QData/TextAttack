from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapMaskedLM


def BERTAttackLi2020(model):
    """
        Li, L.., Ma, R., Guo, Q., Xiangyang, X., Xipeng, Q. (2020).
        
        BERT-ATTACK: Adversarial Attack Against BERT Using BERT
        
        https://arxiv.org/abs/2004.09984
        
        This is "attack mode" 1 from the paper, BAE-R, word replacement.
    """
    from textattack.shared.utils import logger

    logger.warn(
        "WARNING: This BERT-Attack implementation is based off of a"
        " preliminary draft of the paper, which lacked source code and"
        " did not include any hyperparameters. Attack reuslts are likely to"
        " change."
    )
    #
    transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=50)
    #
    # Don't modify the same word twice or stopwords.
    #
    constraints = [RepeatModification(), StopwordModification()]

    # "We only take ε percent of the most important words since we tend to keep
    # perturbations minimum."
    # TODO what is eps?
    constraints.append(MaxWordsPerturbed(max_percent=0.2))

    # "As used in TextFooler (Jin et al., 2019), we also use Universal Sentence
    # Encoder (Cer et al., 2018) to measure the semantic consistency between the
    # adversarial sample and the original sequence. To balance between semantic
    # preservation and attack success rate, we set up a threshold of semantic
    # similarity score to filter the less similar examples."
    #
    # TODO what is the threshold?
    # TODO what window size should be set?
    # TODO should we skip text shorter than the window?
    use_constraint = UniversalSentenceEncoder(
        threshold=0.8,
        metric="cosine",
        compare_with_original=True,
        window_size=15,
        skip_text_shorter_than_window=True,
    )
    constraints.append(use_constraint)
    #
    # Goal us untargeted classification.
    #
    goal_function = UntargetedClassification(model)
    #
    # "We first select the words in the sequence which have a high significance
    # influence on the final output logit. Let S = [w0, ··· , wi ··· ] denote
    # the input sentence, and oy(S) denote the logit output by the target model
    # for correct label y, the importance score Iwi is defined as
    # Iwi = oy(S) − oy(S\wi), where S\wi = [w0, ··· , wi−1, [MASK], wi+1, ···]
    # is the sentence after replacing wi with [MASK]. Then we rank all the words
    # according to the ranking score Iwi in descending order to create word list
    # L."
    search_method = GreedyWordSwapWIR(wir_method="unk")

    return Attack(goal_function, constraints, transformation, search_method)
