.. _constraints:

=============
Constraints
=============

Constraints determine whether a given transformation is valid. Since transformations may not perfectly preserve syntax of semantics, constraints can increase the likelihood that the resulting transformation preserves these qualities. All constraints are subclasses of the constrain abstract class, documeted here, and must implement at least one of ``__call__`` or ``call_many``. 

We split constraints into three main categories.

   :ref:`semantics`: Based on the meaning of input and perturbation

   :ref:`grammaticality`: Based on syntactic properties like part-of-speech and grammar
   
   :ref:`overlap`: Based on character-based properties, like edit distance

.. automodule:: textattack.constraints.constraint
   :members:

.. _semantics:

Semantics
----------

Semantic constraints determine if a transformation is valid based on similarity 
of the semantics between the orignal input and the transformed input.

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
