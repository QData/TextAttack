""".. _constraint:

Constraints
===================

Constraints determine whether a given transformation is valid. Since transformations do not perfectly preserve semantics semantics or grammaticality, constraints can increase the likelihood that the resulting transformation preserves these qualities. All constraints are subclasses of the ``Constraint`` abstract class, and must implement at least one of ``__call__`` or ``call_many``.

We split constraints into three main categories.

   :ref:`Semantics <semantics>`: Based on the meaning of the input and perturbation.

   :ref:`Grammaticality <grammaticality>`: Based on syntactic properties like part-of-speech and grammar.

   :ref:`Overlap <overlap>`: Based on character-based properties, like edit distance.

A fourth type of constraint restricts the search method from exploring certain parts of the search space:

   :ref:`pre_transformation <pre_transformation>`: Based on the input and index of word replacement.
"""

from .pre_transformation_constraint import PreTransformationConstraint
from .constraint import Constraint

from . import grammaticality
from . import semantics
from . import overlap
from . import pre_transformation
