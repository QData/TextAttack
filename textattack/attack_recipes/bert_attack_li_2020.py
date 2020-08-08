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

from .attack_recipe import AttackRecipe


class BERTAttackLi2020(AttackRecipe):
    """Li, L.., Ma, R., Guo, Q., Xiangyang, X., Xipeng, Q. (2020).

    BERT-ATTACK: Adversarial Attack Against BERT Using BERT

    https://arxiv.org/abs/2004.09984

    This is "attack mode" 1 from the paper, BAE-R, word replacement.
    """

    @staticmethod
    def build(model):
        # [from correspondence with the author]
        # Candidate size K is set to 48 for all data-sets.
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        #
        # Don't modify the same word twice or stopwords.
        #
        constraints = [RepeatModification(), StopwordModification()]

        # "We only take ε percent of the most important words since we tend to keep
        # perturbations minimum."
        #
        # [from correspondence with the author]
        # "Word percentage allowed to change is set to 0.4 for most data-sets, this
        # parameter is trivial since most attacks only need a few changes. This
        # epsilon is only used to avoid too much queries on those very hard samples."
        constraints.append(MaxWordsPerturbed(max_percent=0.4))

        # "As used in TextFooler (Jin et al., 2019), we also use Universal Sentence
        # Encoder (Cer et al., 2018) to measure the semantic consistency between the
        # adversarial sample and the original sequence. To balance between semantic
        # preservation and attack success rate, we set up a threshold of semantic
        # similarity score to filter the less similar examples."
        #
        # [from correspondence with author]
        # "Over the full texts, after generating all the adversarial samples, we filter
        # out low USE score samples. Thus the success rate is lower but the USE score
        # can be higher. (actually USE score is not a golden metric, so we simply
        # measure the USE score over the final texts for a comparison with TextFooler).
        # For datasets like IMDB, we set a higher threshold between 0.4-0.7; for
        # datasets like MNLI, we set threshold between 0-0.2."
        #
        # Since the threshold in the real world can't be determined from the training
        # data, the TextAttack implementation uses a fixed threshold - determined to
        # be 0.2 to be most fair.
        use_constraint = UniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification.
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
