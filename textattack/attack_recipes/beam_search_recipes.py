from textattack.shared.attack import Attack
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import BeamSearch
from textattack.transformations import WordSwapEmbedding

def BeamSearch4WordEmbeddingStrong(model):
    transformation = WordSwapEmbedding(max_candidates=50)
    stopwords = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
    constraints = [
        RepeatModification(),
        StopwordModification(stopwords=stopwords)
    ]
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=0.9)
    )

    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=4)

    return Attack(goal_function, constraints, transformation, search_method)

def BeamSearch8WordEmbeddingStrong(model):
    transformation = WordSwapEmbedding(max_candidates=50)
    stopwords = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])
    constraints = [
        RepeatModification(),
        StopwordModification(stopwords=stopwords)
    ]
    constraints.append(
        WordEmbeddingDistance(min_cos_sim=0.9)
    )

    goal_function = UntargetedClassification(model)
    search_method = BeamSearch(beam_width=8)

    return Attack(goal_function, constraints, transformation, search_method)
