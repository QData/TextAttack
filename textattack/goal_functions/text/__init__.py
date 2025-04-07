"""

Goal Function for Text to Text case
---------------------------------------------------------------------

"""

from .minimize_bleu import MinimizeBleu
from .non_overlapping_output import NonOverlappingOutput
from .lev_exceeds_target_dist import LevenshteinExceedsTargetDistance
from .text_to_text_goal_function import TextToTextGoalFunction
from .toxic import Toxic
from .emotion import Emotion
from .ner import Ner
from .mnli import Mnli