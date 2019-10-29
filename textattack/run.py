'''
A command line parser to run an attack

'''

import argparse

parser = argparse.ArgumentParser(description='A commandline parser for TextAttack')

parser.add_argument('--attack', type=str, required=True, 
    choices=['greedy-counterfit', 'ga-counterfit'], help='The type of attack to run')

parser.add_argument('--model', type=str, required=True, 
    choices=['bert-sentiment'], help='The classification model to attack')

parser.add_argument('--constraint', type=float, required=False, 
    help='A constraint to add to the attack')

parser.add_argument('--data', type=str, required=True, 
    help='The dataset to use')

parser.add_argument('-n', type=int, required=False, default=None, 
    help='The number of examples to test on')

args = parser.parse_args()


if __name__ == '__main__':
    import attacks
    import models
    import constraints
    import transformations
    import datasets

    if args.model == 'bert-sentiment':
        model = models.BertForSentimentClassification()
    
    transformation = transformations.WordSwapCounterfit()

    if args.attack == 'greedy-counterfit':
        attack = attacks.GreedyWordSwap(model, transformation)
    elif args.attack == 'ga-counterfit':
        attack = attacks.GeneticAlgorithm(model, transformation)

    if args.constraint:
        attack.add_constraints((
            constraints.semantics.UniversalSentenceEncoder(args.constraint, metric='cosine'),
        ))

    if args.data == 'yelp-sentiment':
        data = datasets.YelpSentiment(args.n)

    attack.attack(data, shuffle=False)
        

