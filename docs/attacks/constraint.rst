.. _constraint:

=============
Constraint
=============

Constraints determine whether a given transformation is valid. Since transformations do not perfectly preserve semantics semantics or grammaticality, constraints can increase the likelihood that the resulting transformation preserves these qualities. All constraints are subclasses of the ``Constraint`` abstract class, and must implement at least one of ``__call__`` or ``call_many``. 

We split constraints into three main categories.

   :ref:`Semantics`: Based on the meaning of the input and perturbation.

   :ref:`Grammaticality`: Based on syntactic properties like part-of-speech and grammar.
   
   :ref:`Overlap`: Based on character-based properties, like edit distance.

A fourth type of constraint restricts the search method from exploring certain parts of the search space:

   :ref:`pre_transformation`: Based on the input and index of word replacement.

.. automodule:: textattack.constraints.constraint
   :special-members: __call__
   :private-members:
   :members:

.. _semantics:

Semantics
----------

Semantic constraints determine if a transformation is valid based on similarity 
of the semantics of the orignal input and the transformed input.

Word Embedding Distance 
########################
.. automodule:: textattack.constraints.semantics.word_embedding_distance
   :members:

Sentence Encoders
##################
.. automodule:: textattack.constraints.semantics.sentence_encoders.sentence_encoder
   :members:

Thought Vectors 
****************
.. automodule:: textattack.constraints.semantics.sentence_encoders.thought_vector
   :members:

BERT 
*****
.. automodule:: textattack.constraints.semantics.sentence_encoders.bert.bert 
   :members:

InferSent 
***********
.. automodule:: textattack.constraints.semantics.sentence_encoders.infer_sent.infer_sent_model
   :members:

.. automodule:: textattack.constraints.semantics.sentence_encoders.infer_sent.infer_sent
   :members:

Universal Sentence Encoder 
***************************
.. automodule:: textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder.universal_sentence_encoder
   :members:


.. _grammaticality:

Grammaticality
-----------------

Grammaticality constraints determine if a transformation is valid based on
syntactic properties of the perturbation.

Language Models
################
.. automodule:: textattack.constraints.grammaticality.language_models.language_model_constraint
   :members:
   
GPT-2
*******

.. automodule:: textattack.constraints.grammaticality.language_models.gpt2
   :members:
   
"Learning To Write" Language Model
************************************

.. automodule:: textattack.constraints.grammaticality.language_models.learning_to_write.learning_to_write
   :members:


Google 1-Billion Words Language Model
**************************************

.. automodule:: textattack.constraints.grammaticality.language_models.google_language_model.google_language_model
   :members:

LanguageTool Grammar Checker 
##############################
.. automodule:: textattack.constraints.grammaticality.language_tool
   :members:

Part of Speech 
###############
.. automodule:: textattack.constraints.grammaticality.part_of_speech
   :members:

.. _overlap:

Overlap
-----------

Overlap constraints determine if a transformation is valid based on character-level analysis.

BLEU Score 
############
.. automodule:: textattack.constraints.overlap.bleu_score
   :members:

chrF Score 
###########
.. automodule:: textattack.constraints.overlap.chrf_score
   :members:

Lenvenshtein Edit Distance  
############################
.. automodule:: textattack.constraints.overlap.levenshtein_edit_distance
   :members:

METEOR Score  
#############
.. automodule:: textattack.constraints.overlap.meteor_score
   :members:

Maximum Words Perturbed  
###########################
.. automodule:: textattack.constraints.overlap.max_words_perturbed
   :members:

.. _pre_transformation:

Pre-Transformation
-------------------------

Pre-transformation constraints determine if a transformation is valid based on 
only the original input and the position of the replacement. These constraints 
are applied before the transformation is even called. For example, these
constraints can prevent search methods from swapping words at the same index
twice, or from replacing stopwords.

Pre-Transformation Constraint
###############################
.. automodule:: textattack.constraints.pre_transformation.pre_transformation_constraint
   :special-members: __call__
   :private-members:
   :members:

Stopword Modification
########################
.. automodule:: textattack.constraints.pre_transformation.stopword_modification
   :members:
   
Repeat Modification
########################
.. automodule:: textattack.constraints.pre_transformation.repeat_modification
   :members:

Input Column Modification
#############################
.. automodule:: textattack.constraints.pre_transformation.input_column_modification
   :members: 
 
Max Word Index Modification
###############################
.. automodule:: textattack.constraints.pre_transformation.max_word_index_modification
   :members:
