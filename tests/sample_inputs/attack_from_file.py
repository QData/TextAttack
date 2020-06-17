import textattack


def Attack(model):
    goal_function = textattack.goal_functions.UntargetedClassification(model)
    search_method = textattack.search_methods.GreedyWordSwapWIR()
    transformation = textattack.transformations.WordSwapRandomCharacterSubstitution()
    constraints = []
    return textattack.shared.Attack(
        goal_function, constraints, transformation, search_method
    )
