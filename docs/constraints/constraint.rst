.. _constraints:

=============
Constraints
=============

Constraints determine whether a given transformation is valid. Since transformations may not perfectly preserve syntax of semantics, constraints can increase the likelihood that the resulting transformation preserves these qualities. All constraints are subclasses of the constrain abstract class, documeted here, and must implement at least one of ``__call__`` or ``call_many``. 

We split constraints into three main categories:

   :ref:`semantic`: Based on the meaning of input and perturbation

   :ref:`grammaticality`: Based on syntactic properties like part-of-speech and grammar
   
   :ref:`overlap`: Based on character-based properties, like edit distance

.. automodule:: textattack.constraints.constraint
   :members:

