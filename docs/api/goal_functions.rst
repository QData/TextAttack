Goal Functions API Reference
============================

:class:`~textattack.goal_functions.GoalFunction` determines both the conditions under which the attack is successful (in terms of the model outputs)
and the heuristic score that we want to maximize when searching for the solution.

GoalFunction
------------
.. autoclass:: textattack.goal_functions.GoalFunction
   :members:

ClassifiedGoalFunction
------------
.. autoclass:: textattack.goal_functions.ClassifiedGoalFunction
   :members:

TargetedClassification
------------
.. autoclass:: textattack.goal_functions.TargetedGoalFunction
   :members:

UntargetedClassification
------------
.. autoclass:: textattack.goal_functions.UntargetedGoalFunction
   :members:

InputReduction
------------
.. autoclass:: textattack.goal_functions.InputReduction
   :members:

TextToTextGoalFunction
------------
.. autoclass:: textattack.goal_functions.TextToTextGoalFunction
   :members:

MinimizeBleu
------------
.. autoclass:: textattack.goal_functions.MinimizeBleu
   :members:

NonOverlappingOutput(
------------
.. autoclass:: textattack.goal_functions.NonOverlappingOutput(
   :members:

