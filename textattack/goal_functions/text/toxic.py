import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction

class Toxic(TextToTextGoalFunction):
    """This is a targeted attack on toxic classification models.
    The model emits a series of probabilities for various toxic labels.
    We aim to minimise the sum of all probabilities (i.e. one should start with an originally toxic sentence 
    and attempt to perturb it such that the model think it is not toxic.)
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, _)
        if (score == 0.0):
            return True
        else:
            return False

    def _get_score(self, model_output, _):
        # the model output looks like this: {'toxic': 0.9773269, 'severe_toxic': 0.001563263, 'obscene': 0.028068777, 'threat': 0.0014984279, 'insult': 0.550452, 'identity_hate': 0.006552372}
        # We aim to minimise the sum of all labels
        return sum(model_output.values())

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_distance"]
