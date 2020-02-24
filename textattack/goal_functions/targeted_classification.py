from textattack.goal_functions import GoalFunction

class TargetedClassification(GoalFunction):
   
    def __init__(self, model, target_class=None):
        super.__init__(model)
        if target_class:
            self.target_class = target_class

    def set_original_attrs(self, tokenized_text, correct_output):
        super.set_original_attrs(tokenized_text, correct_output)
        if not target_class:
            self.target_class = 1 if correct_output == 0 else 0

    def _is_goal_complete(self, model_output):
        return model_output.argmax() == self.target_class

    def _get_score(self, model_output):
        return model_output[self.target_class]
        
    def _get_output(self, raw_output):
        return raw_output.argmax()

