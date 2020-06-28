from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapMaskedLM


def BAEGarg2019(model):
    """
        Siddhant Garg and Goutham Ramakrishnan, 2019.
        
        BAE: BERT-based Adversarial Examples for Text Classification.
        
        https://arxiv.org/pdf/2004.01970
        
        This is "attack mode" 1 from the paper, BAE-R, word replacement.
        
        We present 4 attack modes for BAE based on the
            R and I operations, where for each token t in S:
            • BAE-R: Replace token t (See Algorithm 1)
            • BAE-I: Insert a token to the left or right of t
            • BAE-R/I: Either replace token t or insert a
            token to the left or right of t
            • BAE-R+I: First replace token t, then insert a
            token to the left or right of t
    """
    # "In this paper, we present a simple yet novel technique: BAE (BERT-based
    # Adversarial Examples), which uses a language model (LM) for token
    # replacement to best fit the overall context. We perturb an input sentence
    # by either replacing a token or inserting a new token in the sentence, by
    # means of masking a part of the input and using a LM to fill in the mask."
    #
    # We only consider the top K=50 synonyms from the MLM predictions.
    #
    # [from email correspondance with the author]
    # "When choosing the top-K candidates from the BERT masked LM, we filter out
    # the sub-words and only retain the whole words (by checking if they are
    # present in the GloVE vocabulary)"
    #
    transformation = WordSwapMaskedLM(method="bae", max_candidates=50)
    #
    # Don't modify the same word twice or stopwords.
    #
    constraints = [RepeatModification(), StopwordModification()]

    # For the R operations we add an additional check for
    # grammatical correctness of the generated adversarial example by filtering
    # out predicted tokens that do not form the same part of speech (POS) as the
    # original token t_i in the sentence.
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

    # To ensure semantic similarity on introducing perturbations in the input
    # text, we filter the set of top-K masked tokens (K is a pre-defined
    # constant) predicted by BERT-MLM using a Universal Sentence Encoder (USE)
    # (Cer et al., 2018)-based sentence similarity scorer.
    #
    # [We] set a threshold of 0.8 for the cosine similarity between USE-based
    # embeddings of the adversarial and input text.
    #
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
    # "We estimate the token importance Ii of each token
    # t_i ∈ S = [t1, . . . , tn], by deleting ti from S and computing the
    # decrease in probability of predicting the correct label y, similar
    # to (Jin et al., 2019).
    #
    # [Note that this isn't what (Jin et al., 2019) did, since the WIR method
    # is `delete` instead of `unk`.]
    #
    # • "If there are multiple tokens can cause C to misclassify S when they
    # replace the mask, we choose the token which makes Sadv most similar to
    # the original S based on the USE score."
    # • "If no token causes misclassification, we choose the perturbation that
    # decreases the prediction probability P(C(Sadv)=y) the most."
    #
    search_method = GreedyWordSwapWIR(wir_method="delete")

    return Attack(goal_function, constraints, transformation, search_method)