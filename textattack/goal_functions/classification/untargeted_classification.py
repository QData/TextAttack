from .classification_goal_function import ClassificationGoalFunction

class UntargetedClassification(ClassificationGoalFunction):
    """
    An untargeted attack on classification models which attempts to minimize the 
    score of the correct label until it is no longer the predicted label.
    """
    
    def _is_goal_complete(self, model_output, correct_output):
        import pdb; pdb.set_trace()
        return model_output.argmax() != correct_output 

    def _get_score(self, model_output, correct_output):
        return -model_output[correct_output]

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())
        
