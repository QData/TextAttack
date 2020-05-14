.. _constraints:

=============
Constraints
=============

Constraints determine whether a given transformation is valid. Since transformations may not perfectly preserve syntax of semantics, constraints can increase the likelihood that the resulting transformation preserves these qualities. All constraints are subclasses of the constrain abstract class, documeted here, and must implement at least one of ``__call__`` or ``call_many``. 

We split constraints into three main categories:

   :ref:`semantics`: Check meaning of sentence

   :ref:`syntactical`: Check part-of-speech and grammar
   
   :ref:`overlap`: Measure edit distance

.. automodule:: textattack.constraints.constraint
   :members:

