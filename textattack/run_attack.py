'''
A command line parser to run an attack

'''

import argparse

parser = argparse.ArgumentParser(description='A commandline parser for TextAttack', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

data_group = parser.add_mutually_exclusive_group(required=True)

parser.add_argument('--attack', type=str, required=False, default='greedy-wir-counterfit',
    choices=['greedy-counterfit', 'ga-counterfit', 'greedy-wir-counterfit'], 
    help='The type of attack to run.')

parser.add_argument('--model', type=str, required=False, default='bert-sentiment',
    choices=['bert-sentiment'], help='The classification model to attack.')

parser.add_argument('--transformation', type=str, required=False, default='word-swap-embedding',
    choices=['word-swap-embedding', 'word-swap-homoglyph'],
    help='The type of transformation to apply')

parser.add_argument('--constraints', type=str, required=False, nargs='*',
    default=['use', 'lang-tool'], 
    help=('Constraints to add to the attack. Usage: "--constraints use lang-tool:{threshold} ' 
    'goog-lm" to use the default use similarity of .9 and to choose the threshold'))

parser.add_argument('--out_file', type=str, required=False,
    help='The file to output the results to.')

parser.add_argument('--num_examples', type=int, required=False, 
    help='The number of examples to attack.')

data_group.add_argument('--interactive', action='store_true', 
    help='Whether to run attacks interactively.')

data_group.add_argument('--data', type=str,
    choices=['yelp-sentiment'], help='The dataset to use.')

args = parser.parse_args()


if __name__ == '__main__':
    import attacks
    import models
    import constraints
    import transformations
    import datasets
    import time
    import os
    from tokenized_text import TokenizedText

    # Only use one GPU, if we have one.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable tensorflow logs, except in the case of an error.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start_time = time.time()

    #Models
    if args.model == 'bert-sentiment':
        model = models.BertForSentimentClassification()
    
    #Transformation
    if args.transformation == 'word-swap-embedding':
        transformation = transformations.WordSwapEmbedding()
    elif args.transformation == 'word-swap-homoglyph':
        transformation = transformations.WordSwapHomoglyph()

    #Attacks
    if args.attack == 'greedy-counterfit':
        attack = attacks.GreedyWordSwap(model, transformation)
    elif args.attack == 'ga-counterfit':
        attack = attacks.GeneticAlgorithm(model, transformation)
    elif args.attack == 'greedy-wir-counterfit':
        attack = attacks.GreedyWordSwapWIR(model, transformation)

    #Constraints
    if args.constraints:

        defined_constraints = []

        for constraint in args.constraints:
            if 'use:' in constraint:
                similarity = constraint.replace('use:', '')
                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(float(similarity), metric='cosine'))
            elif constraint == 'use':
                #Default similarity to .90 if no similarity is given
                defined_constraints.append(constraints.semantics.UniversalSentenceEncoder(.90, metric='cosine'))

            elif 'lang-tool:' in constraint:
                threshold = constraint.replace('lang-tool:', '')
                defined_constraints.append(constraints.syntax.LanguageTool(float(threshold)))
            elif constraint == 'lang-tool':
                #Default threshold to 1 if no threshold is given
                defined_constraints.append(constraints.syntax.LanguageTool(1))

            elif constraint == 'goog-lm':
                defined_constraints.append(constraints.semantics.google_language_model.GoogleLanguageModel())

            else:
                raise ValueError((f'{constraint} is not a valid constraint. ' 
                    'Valid options are "use", "lang-tool", or "goog-lm". Use "-h" for help.'))

        attack.add_constraints(defined_constraints)

    #Data
    if args.data == 'yelp-sentiment':
        data = datasets.YelpSentiment(args.num_examples)

    #Output file
    if args.out_file is not None:
        attack.add_output_file(args.out_file)


    load_time = time.time()

    if args.data is not None:
        attack.attack(data, shuffle=False)

        finish_time = time.time()

        print(f'Loaded in {load_time - start_time}s')
        print(f'Ran attack in {finish_time - load_time}s')
        print(f'TOTAL TIME: {finish_time - start_time}s')

    
    if args.interactive:
        print('Running in interactive mode')
        print('----------------------------')

        while True:
            print('Enter a sentence to attack or "q" to quit:')
            text = input()

            if text == 'q':
                break

            tokenized_text = TokenizedText(model, text)

            pred = attack._call_model([tokenized_text])
            label = int(pred.argmax())

            print('Attacking...')

            attack.attack([(label, text)])

