===========
Overview
===========
TextAttack builds attacks from four components: a search method, goal function, transformation, and a set of constraints. 

- A **search method** explores the transformation space and attempts to find a successful attack as determined by a set of constraints. 
- A **goal function** determines if an attack has succeeded.
- A **transformation** perturbs the text input.
- **Constraints** determine if a transformation is successful. 

Any model that overrides ``__call__``, takes ``TokenizedText`` as input, and formats output correctly can be used with TextAttack. TextAttack also has built-in datasets and pre-trained models on these datasets. Below is an example of attacking a pre-trained model on the AGNews dataset::

    from tqdm import tqdm
    from textattack.loggers import FileLogger

    from textattack.datasets.classification import AGNews
    from textattack.models.classification.lstm import LSTMForAGNewsClassification
    from textattack.goal_functions import UntargetedClassification

    from textattack.search_methods import GeneticAlgorithm
    from textattack.transformations import WordSwapEmbedding
    from textattack.constraints.grammaticality import PartOfSpeech

    # Create the model and goal function
    model = LSTMForAGNewsClassification()
    goal_function = UntargetedClassification(model)

    # Use the default WordSwapEmbedding transformation 
    transformation = WordSwapEmbedding()

    # Add a constraint, note that an empty list can be used if no constraints are wanted
    constraints = [PartOfSpeech()]

    # Make an attack with the above parameters
    attack = GeneticAlgorithm(goal_function, transformation, constraints)

    # Run the attack on 5 examples and see the results using a logger to output to stdout
    results = attack.attack_dataset(AGNews(), num_examples=5, attack_n=True)

    logger = FileLogger(stdout=True)

    for result in tqdm(results, total=5): 
        logger.log_attack_result(result)
    


For more examples and information, see our examples on GitHub

- `A custom transformation <https://github.com/QData/TextAttack/blob/master/examples/%5B1%5D%20Introduction%20%26%20Transformations.ipynb>`__
- `A custom constraint <https://github.com/QData/TextAttack/blob/master/examples/%5B2%5D%20Constraints.ipynb>`__

