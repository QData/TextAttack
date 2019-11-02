'''
A command line parser to run an attack

'''

import argparse

parser = argparse.ArgumentParser(description='A commandline parser for TextAttack')

parser.add_argument('--attack', type=str, required=True, 
    choices=['greedy-counterfit', 'ga-counterfit', 'greedy-wir'], help='The type of attack to run')

parser.add_argument('--model', type=str, required=True, 
    choices=['bert-sentiment', 'infer-sent'], help='The classification model to attack')

parser.add_argument('--constraints', type=str, required=False, nargs='*',
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
    import time

    start_time = time.time()

    #Models
    if args.model == 'bert-sentiment':
        model = models.BertForSentimentClassification()
    elif args.model == 'infer-sent':
        #Doesn't work for now
        raise NotImplementedError()
        model = models.InferSent()
    
    #Transformation
    transformation = transformations.WordSwapEmbedding()

    #Attacks
    if args.attack == 'greedy-counterfit':
        attack = attacks.GreedyWordSwap(model, transformation)
    elif args.attack == 'ga-counterfit':
        attack = attacks.GeneticAlgorithm(model, transformation)
    elif args.attack == 'greedy-wir':
        attack = attacks.GreedyWordSwapWIR(model, transformation)

    #Constraints
    if args.constraints:

        defined_constraints = []

        for constraint in args.constraints:
            if 'sim:' in constraint:
                similarity = constraint.replace('sim:', '')

                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(float(similarity), metric='cosine'),)

                # attack.add_constraints(
                #     (
                #     constraints.semantics.UniversalSentenceEncoder(float(similarity), metric='cosine'),
                #     )
                # )
            elif 'lang-tool' in constraint:
                threshold = constraint.replace('lang-tool:', '')

                defined_constraints.append(constraints.syntax.LanguageTool(float(threshold)),)

                # attack.add_constraints(
                #     (
                #     constraints.syntax.LanguageTool(float(threshold)),
                #     )
                # )

        attack.add_constraints(defined_constraints)

    #Data
    if args.data == 'yelp-sentiment':
        data = datasets.YelpSentiment(args.n)


    load_time = time.time()

    attack.attack(data, shuffle=False)

    finish_time = time.time()

    print(f'Loaded in {load_time - start_time}s')
    print(f'Ran attack in {finish_time - load_time}s')
    print(f'TOTAL TIME: {finish_time - start_time}s')
        

