"""
Imperceptible Perturbations Algorithm
========================

"""

from .attack_recipe import AttackRecipe
from textattack.goal_functions import Emotion, Mnli, Ner, Toxic, MaximizeLevenshtein, MinimizeBleu
from textattack.transformations import WordSwapInvisibleCharacters, WordSwapHomoglyphSwap, WordSwapDeletions, WordSwapReorderings
from textattack.search_methods import DifferentialEvolution
from textattack import Attack


class ImperceptibleRecipe(AttackRecipe):

    """
    This attack recipe can be used to create some of the attacks outlined in the https://arxiv.org/abs/2106.09898 paper.

    It attacks a model with an imperceptible transformation and uses the search method 
    textattack.search_methods.DifferentialEvolution, an implementation of differential evolution.

    The imperceptible transformations supported are as follows:
    - textattack.transformations.WordSwapInvisibleCharacters: 
        - Injects (perturbs) number of invisible characters into the input string
        - Specified with perturbation_type = "invisible"
    - textattack.transformations.WordSwapHomoglyphSwap:
        - Replaces (perturbs) number of characters in the input string with homoglyphs
        - Specified with perturbation_type = "hommoglyphs"
    - textattack.transformations.WordSwapDeletions:
        - Injects (perturbs) number of deletion control characters into the input string
        - Specified with perturbation_type = "deletions"
    - textattack.transformations.WordSwapReorderings: 
        - Injects (perturbs) number of sets of reordering control characters into the input string
        - Specified with perturbation_type = "reorderings"

    The attacks supported are as follows:

    (1) Targeted classification attack on a sentiment analysis model
    - This is called with task_type = "emotion"
    - It requires a model wrapper that takes as input a List[str], and returns a list of elements, one per input string,
    where each element is a list/numpy array/tensor of probabilities for each sentiment
    - An example of a compatible model wrapper is 
        textattack.models.wrappers.EmotionWrapper
    initialised with the model
        pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=-1)

    (2) Targeted classification attack on a natural language inference model
    - This is called with task_type = "mnli"
    - It requires a model wrapper that takes as input a List[Tuple[str, str]], and returns a list of elements, one per input string, 
    where each element is a list/numpy array/tensor of probabilities for each possible outcome
    The size of each element does not matter, but it is usually 3 ([contradiction, neutral, entailment])
    - An example of a compatible model wrapper is 
        textattack.models.wrappers.FairseqMnliWrapper
    initialised with the model
        torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').eval()

    (3) Token-level targeted classification attack on a named entity recognition model
    - This is called with task_type = "ner"
    - It requires a model wrapper that takes as input a List[str], and returns a list of elements, one per input string,
    where each element is a list of dictionaries, where each dictionary must have the keys 'entity' and 'score'.
    An example element is {'entity': 'I-MISC', 'score': 0.99509996, 'index': 6, 'word': 'J', 'start': 8, 'end': 9}
    - An example of a compatible model wrapper is 
        textattack.models.wrappers.PipelineModelWrapper
    initialised with the model
        pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    (4) Untargeted classification attack on a toxic classifier model
    - This is called with task_type = "toxic"
    - It requires a model wrapper that takes as input a List[str], and returns a list of elements, one per input string, 
    where each element is a list/numpy array/tensor of logits, one for each possible outcome
    - An example of a compatible model wrapper is
        textattack.models.wrappers.IBMMaxTopicWrapper
    initialised with the ModelWrapper() at
        https://github.com/IBM/MAX-Toxic-Comment-Classifier/blob/master/core/model.py 

    (5) Untargeted text-to-text attacks on a machine translation model 
    - This is called with
        task_type = "translation_bleu" if the desired goal is to maximize Bleu score between actual and perturbed output
        task_type = "translation_levenshtein" if the desired goal is to maximize Levenshtein distance between actual and perturbed output
    - It requires a model wrapper that takes as input a List[str], and returns a list of elements, one per input string,
    where each element is a single string, the result of the translation
    - An example of a compatible model wrapper is
        textattack.models.wrappers.FairseqTranslationWrapper
    initialised with the model
        torch.hub.load('pytorch/fairseq',
                        'transformer.wmt14.en-fr',
                        tokenizer='moses',
                        bpe='subword_nmt',
                        verbose=False).eval()

    All "perturbation_type"s are compatible with all "task_type"s.

    Note: It is possible to use other goal functions such as textattack.goal_functions.UntargetedClassfication with the imperceptible transformations.
    These are not implemented in this recipe. It is important to ensure that the search method and model wrapper chosen are compatible with the goal function.
    For example, simply using DifferentialEvolution on UntargetedClassification may return unexpected results because the custom goal functions we have written
    override the _get_score method in UntargetedClassification to return something sensible for DifferentialEvolution's objective function.

    """

    @staticmethod
    def build(model_wrapper, task_type: str, perturbation_type: str, perturbs=1, popsize=32, maxiter=10):
        """
        Args:
            model_wrapper: Ensure compatibility with task_type (see above)
            task_type: One of "emotion", "mnli", "ner", "toxic", "translation_bleu" or "translation_levenshtein".
            perturbation_type: One of "homoglyphs", "invisible", "deletions" or "reorderings".
            perturbs: Number of perturbations to the input string allowed. Default = 1.
            popsize, maxiter: Parameters for differential evolution. Default = 32, 10, as implemented in the paper.
        """

        if task_type == "emotion":
            goal_function = Emotion(model_wrapper)
        elif task_type == "mnli":
            goal_function = Mnli(model_wrapper)
        elif task_type == "ner":
            goal_function = Ner(model_wrapper)
        elif task_type == "toxic":
            goal_function = Toxic(model_wrapper, target_max_score=0.5)
        elif task_type == "translation_bleu":
            goal_function = MinimizeBleu(model_wrapper)
        elif task_type == "translation_levenshtein":
            goal_function = MaximizeLevenshtein(model_wrapper)
        
        if perturbation_type == "homoglyphs":
            transformation = WordSwapHomoglyphSwap()
        elif perturbation_type == "invisible":
            transformation = WordSwapInvisibleCharacters()
        elif perturbation_type == "deletions":
            transformation = WordSwapDeletions()
        elif perturbation_type == "reorderings":
            transformation = WordSwapReorderings
        
        search_method = DifferentialEvolution(
            popsize=popsize, 
            maxiter=maxiter, 
            verbose=True,
            max_perturbs=perturbs
        )

        constraints = []

        return Attack(goal_function, constraints, transformation, search_method)

        
            
