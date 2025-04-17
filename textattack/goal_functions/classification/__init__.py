"""

Goal fucntion for Classification
---------------------------------------------------------------------

"""

from .input_reduction import InputReduction
from .classification_goal_function import ClassificationGoalFunction
from .untargeted_classification import UntargetedClassification
from .targeted_classification import TargetedClassification

from .toxic import Toxic
from .mnli import Mnli
from .emotion import Emotion
from .ner import Ner

from .unnormalized_classification import UnnormalizedClassification
from .unprocessed_classification import UnprocessedClassification