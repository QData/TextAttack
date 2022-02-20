Goal Functions API Reference
============================

:class:`~textattack.goal_functions.GoalFunction` determines both the conditions under which the attack is successful (in terms of the model outputs)
and the heuristic score that we want to maximize when searching for the solution.

GoalFunction
------------
.. autoclass:: textattack.goal_functions.GoalFunction
   :members:

ClassificationGoalFunction
--------------------------
.. autoclass:: textattack.goal_functions.classification.ClassificationGoalFunction
   :members:

TargetedClassification
----------------------
.. autoclass:: textattack.goal_functions.classification.TargetedClassification
   :members:

UntargetedClassification
------------------------
.. autoclass:: textattack.goal_functions.classification.UntargetedClassification
   :members:

InputReduction
--------------
.. autoclass:: textattack.goal_functions.classification.InputReduction
   :members:

TextToTextGoalFunction
-----------------------
.. autoclass:: textattack.goal_functions.text.TextToTextGoalFunction
   :members:

MinimizeBleu
-------------
.. autoclass:: textattack.goal_functions.text.MinimizeBleu
   :members:

NonOverlappingOutput
----------------------
.. autoclass:: textattack.goal_functions.text.NonOverlappingOutput
   :members:

