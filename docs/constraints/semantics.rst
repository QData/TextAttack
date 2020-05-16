.. _semantics:

================================
Semantics
================================

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

