from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import sys
from InquirerPy import prompt
from InquirerPy.exceptions import InvalidArgument
from InquirerPy.validator import PathValidator
from pyfiglet import Figlet

from eukaryote import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from eukaryote.commands import TextAttackCommand
from eukaryote.commands.t4a_attack_eval_command import T4A_AttackEvalCommand
from eukaryote.commands.t4a_train_command import T4A_TrainCommand
from eukaryote.commands.t4a_attack_train_command import T4A_AttackTrainCommand
import eukaryote.t4a.shared as shared


class InteractiveCliCommand(TextAttackCommand):

    def run(self, args):
        f = Figlet(font='slant')
        print(f.renderText('Eukaryote Interactive'))

        action_choices = ['Attack a model with attack_eval', 'Adversarial training with attack_train', 'Train a model with train']
        model_choices = ['distilbert-base-uncased', 'BERT_mini', 'BERT_regular', 'BERT_large', 'BagOfWords']
        dataset_choices = ['rotten_tomatoes']
        attack_choices = ['textfooler']

        questions = [
            {
                'message': 'Select an action:',
                'type': 'list',
                'choices': action_choices,
                'name': 'action',
            },
            {
                'message': 'Select a model:',
                'type': 'fuzzy',
                'choices': model_choices,
                'name': 'model_huggingface',
            },
            {
                'message': 'Select a dataset:',
                'type': 'fuzzy',
                'choices': dataset_choices,
                'name': 'dataset_huggingface',
            },
            {
                'message': 'Select an attack recipe:',
                'type': 'fuzzy',
                'choices': attack_choices,
                'name': 'attack_recipe',
            },
            {'message': 'Confirm?', 'type': 'confirm', 'default': False, 'name':'confirmation'},
        ]

        try:
            arguments = prompt(questions, vi_mode=True) # decide which action to take
            if arguments['action'] == 'Attack a model with attack_eval':
                parser_fake = ArgumentParser() # copy t4a_attack_eval_command.py's register_subcommand behavior
                shared.add_arguments_model(parser_fake)
                shared.add_arguments_dataset(parser_fake, default_split="test")
                shared.add_arguments_attack(parser_fake)
                parser_fake.set_defaults(attack=True)
                newArgs = parser_fake.parse_args([f"--{k}={v}" for k, v in arguments.items() if k != 'action' and k != 'confirmation'])
                T4A_AttackEvalCommand().run(args=newArgs)
            elif arguments['action'] == 'Train a model with train': 
                parser_fake = ArgumentParser() # copy t4a_train_command.py's register_subcommand behavior
                shared.add_arguments_model(parser_fake)
                shared.add_arguments_dataset(parser_fake, default_split="train")
                shared.add_arguments_train(parser_fake, default_split_eval="test")
                parser_fake.set_defaults(attack=False)
                newArgs = parser_fake.parse_args([f"--{k}={v}" for k, v in arguments.items() if k != 'action' and k != 'confirmation' and k != 'attack_recipe'])
                T4A_TrainCommand().run(args=newArgs)
            elif arguments['action'] == 'Adversarial training with attack_train': 
                parser_fake = ArgumentParser() # copy t4a_attack_train_command.py's register_subcommand behavior
                shared.add_arguments_model(parser_fake)
                shared.add_arguments_dataset(parser_fake, default_split="train")
                shared.add_arguments_attack(parser_fake)
                shared.add_arguments_train(parser_fake)
                parser_fake.set_defaults(attack=True)
                newArgs = parser_fake.parse_args([f"--{k}={v}" for k, v in arguments.items() if k != 'action' and k != 'confirmation'])
                T4A_AttackTrainCommand().run(args=newArgs)
        except InvalidArgument:
            print('No available choices')

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser_interactive = main_parser.add_parser(
            'interactive',
            help='start the interactive CLI to run a T4A command',
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser_interactive.set_defaults(func=InteractiveCliCommand())