===========
Overview
===========
TextAttack builds attacks from four components:

- `Goal Functions <../attacks/goal_function.html>`__ stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- `Constraints <../attacks/constraint.html>`__ determine if a potential perturbation is valid with respect to the original input.
- `Transformations <../attacks/transformation.html>`__ take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- `Search Methods <../attacks/search_method.html>`__ explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

Any model that overrides ``__call__``, takes ``TokenizedText`` as input, and formats output correctly can be used with TextAttack. TextAttack also has built-in datasets and pre-trained models on these datasets. Below is an example of attacking a pre-trained model on the AGNews dataset::

    from tqdm import tqdm
    from textattack.loggers import FileLogger
    
    from textattack.datasets.classification import AGNews
    from textattack.models.classification.lstm import LSTMForAGNewsClassification
    from textattack.goal_functions import UntargetedClassification
    
    from textattack.shared import Attack
    from textattack.search_methods import GreedySearch
    from textattack.transformations import WordSwapEmbedding
    from textattack.constraints.grammaticality import PartOfSpeech
    from textattack.constraints.semantics import RepeatModification, StopwordModification
    
    # Create the model and goal function
    model = LSTMForAGNewsClassification()
    goal_function = UntargetedClassification(model)
    
    # Use the default WordSwapEmbedding transformation 
    transformation = WordSwapEmbedding()
    
    # Add a constraint, note that an empty list can be used if no constraints are wanted
    constraints = [
        RepeatModification(),
        StopwordModification(),
        PartOfSpeech()
    ]
    
    # Choose a search method
    search = GreedySearch()
    
    # Make an attack with the above parameters
    attack = Attack(goal_function, constraints, transformation, search)
    
    # Run the attack on 5 examples and see the results using a logger to output to stdout
    results = attack.attack_dataset(AGNews(), num_examples=5, attack_n=True)
    
    logger = FileLogger(stdout=True)
    
    for result in tqdm(results, total=5): 
        logger.log_attack_result(result)
