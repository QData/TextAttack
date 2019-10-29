=========
Examples
=========


BERT Example 
############

.. parsed-literal::
   model = BertForSentimentClassification()

   transformation = WordSwapCounterfit()

   transformation.add_constraints(
      constraints.semantics.UniversalSentenceEncoder(0.9, metric='cosine'),
   )

   attack = attacks.GreedyWordSwap(model, transformation)

   yelp_data = YelpSentiment(n=2)

   attack.add_output_file(open('outputs/test.txt', 'w'))

   attack.attack(yelp_data, shuffle=False)