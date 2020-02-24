from textattack.goal_functions import GoalFunction

class UntargetedClassification(GoalFunction):
    
    def _is_goal_complete(self, model_output):
        return model_output.argmax() != self.correct_output    

    def _get_score(self, model_output):
        return -model_output[self.correct_output]

    def _get_output(self, raw_output):
        return raw_output.argmax()
        
