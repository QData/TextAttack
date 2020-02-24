from textattack.goal_functions import GoalFunction

class TargetedClassification(GoalFunction):
   
    def __init__(self, model, target_class=0):
        super().__init__(model)
        self.target_class = target_class

    def _is_goal_complete(self, model_output):
        return self.correct_output != model_output.argmax() or self.correct_output == self.target_class 

    def _get_score(self, model_output):
        return model_output[self.target_class]
        
    def _get_output(self, raw_output):
        return int(raw_output.argmax())

