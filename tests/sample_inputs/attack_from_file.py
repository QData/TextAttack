import eukaryote


def Attack(model):
    goal_function = eukaryote.goal_functions.UntargetedClassification(model)
    search_method = eukaryote.search_methods.GreedyWordSwapWIR()
    transformation = eukaryote.transformations.WordSwapRandomCharacterSubstitution()
    constraints = []
    return eukaryote.Attack(goal_function, constraints, transformation, search_method)
