from textattack.shared.attack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import *
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet
import copy

stopwords = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
BASIC_CONSTRAINTS = [RepeatModification(), StopwordModification(stopwords=stopwords)]

##############################################################################
########################### Word Embedding Distance ##########################
##############################################################################

COSINE_WEAK = 0.25
COSINE_MED = 0.5
COSINE_STRICT = 0.9
WED_transformation = WordSwapEmbedding(max_candidates=50)

def Greedy_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def GreedyWIR_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch4_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch8_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MHA_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = MetropolisHastingsSampling()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def Genetic_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = GeneticAlgorithm()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MCTS_WED_Weak(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_WEAK)
    )
    goal_function = UntargetedClassification(model)
    search_method = MonteCarloTreeSearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)

#########################################################################################################

def Greedy_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def GreedyWIR_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch4_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch8_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MHA_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = MetropolisHastingsSampling()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def Genetic_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = GeneticAlgorithm()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MCTS_WED_Med(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_MED)
    )
    goal_function = UntargetedClassification(model)
    search_method = MonteCarloTreeSearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)

#########################################################################################################

def Greedy_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def GreedyWIR_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch4_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def BeamSearch8_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MHA_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = MetropolisHastingsSampling()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def Genetic_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = GeneticAlgorithm()
    return Attack(goal_function, constraints, WED_transformation, search_method)

def MCTS_WED_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=COSINE_STRICT)
    )
    goal_function = UntargetedClassification(model)
    search_method = MonteCarloTreeSearch()
    return Attack(goal_function, constraints, WED_transformation, search_method)


##############################################################################
################################### WordNet ##################################
##############################################################################

#Angular simliarity
USE_SIM_LAX = 0.7 
USE_SIM_STRICT = 0.925
WORDNET_transformation = WordSwapWordNet()

##############################################################################
def Greedy_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def GreedyWIR_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def BeamSearch4_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def BeamSearch8_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def MHA_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = MetropolisHastingsSampling()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def Genetic_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GeneticAlgorithm()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def MCTS_WordNet_Lax(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_LAX,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = MonteCarloTreeSearch()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

##############################################################################
def Greedy_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def GreedyWIR_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def BeamSearch4_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def BeamSearch8_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def MHA_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = MetropolisHastingsSampling()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def Genetic_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = GeneticAlgorithm()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)

def MCTS_WordNet_Strict(model):
    constraints = copy.deepcopy(BASIC_CONSTRAINTS)
    constraints.append(
        UniversalSentenceEncoder(threshold=USE_SIM_STRICT,
            metric='angular', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    )
    goal_function = UntargetedClassification(model)
    search_method = MonteCarloTreeSearch()
    return Attack(goal_function, constraints, WORDNET_transformation, search_method)