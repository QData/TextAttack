from .unnormalized_classification import UnnormalizedClassification
from .untargeted_classification import UntargetedClassification

class Toxic(UnnormalizedClassification, UntargetedClassification):

    def _get_score(self, model_output, _):
        """
        model_output is a tensor of logits, one for each label.
        
        When used with ImperceptibleDE, this method allows us to minimize the sum of the logits.
        """
        return sum(model_output)

    def _get_displayed_output(self, raw_output):
        return raw_output.tolist()
