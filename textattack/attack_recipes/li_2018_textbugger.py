"""
    Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).
    
    TextBugger: Generating Adversarial Text Against Real-world Applications.
 
    ArXiv, abs/1812.05271..
    
"""

from textattack.attack_methods import GreedyWordSwapWIR
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.syntax import PartOfSpeech
from textattack.transformations import WordSwapEmbedding
from textattack.goal_functions import UntargetedClassification

def Gao2018DeepWordBug(model):
    #
    #  we propose five bug generation methods for TEXTBUGGER:
    #
    # We propose four similar methods:
    transformation = CompositeTransformation([
        # (1) Insert: Insert a space into the word3
        #. Generally, words are segmented by spaces in English. Therefore,
        # we can deceive classifiers by inserting spaces into words.

 #     i(2)
# Delete: Delete a random character of the word except for
# the first and the last character


# (3) Swap: Swap random two
# adjacent letters in the word but do not alter the first or last
# letter4
# . This is a common occurrence when typing quickly
# and is easy to implement. 


# (4) Substitute-C (Sub-C): Replace
# characters with visually similar characters (e.g., replacing “o”
# with “0”, “l” with “1”, “a” with “@”) or adjacent characters in
# the keyboard (e.g., replacing “m” with “n”).

# (5) Substitute-W
# (Sub-W): Replace a word with its topk nearest neighbors in a
    ])
    #
    # In these experiments, we hold the maximum difference
    # on edit distance (ϵ) to a constant 30 for each sample.
    #
    constraints = [
        LevenshteinEditDistance(30)
    ]
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GreedyWordSwapWIR(goal_function, transformation=transformation,
        constraints=constraints, max_depth=None)
    
    return attack
