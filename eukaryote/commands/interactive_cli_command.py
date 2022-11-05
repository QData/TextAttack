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
        f = Figlet(font="slant")
        print(f.renderText("Eukaryote Interactive"))

        action_choices = [
            "Attack a model with attack_eval",
            "Adversarial training with attack_train",
            "Train a model with train",
        ]
        model_choices = [
            "distilbert-base-uncased",
            "BERT_mini",
            "BERT_regular",
            "BERT_large",
            "BagOfWords",
        ]
        dataset_choices = ["rotten_tomatoes"]
        attack_choices = ["textfooler"]

        questions = [
            {
                "message": "Select an action:",
                "type": "list",
                "choices": action_choices,
                "name": "action",
            },
            {
                "message": "Select a model:",
                "type": "fuzzy",
                "choices": model_choices,
                "name": "model_huggingface",
            },
            {
                "message": "Select a dataset:",
                "type": "fuzzy",
                "choices": dataset_choices,
                "name": "dataset_huggingface",
            },
            {
                "message": "Select an attack recipe:",
                "type": "fuzzy",
                "choices": attack_choices,
                "name": "attack_recipe",
            },
            {
                "message": "Confirm?",
                "type": "confirm",
                "default": False,
                "name": "confirmation",
            },
        ]

        # semi-redundant code here, but I have a feeling keeping them separate is more clear+extensible?
        try:
            prompt_action = prompt(questions[0], vi_mode=True)  # decide which action to take
            if prompt_action["action"] == "Attack a model with attack_eval":
                prompt_args = prompt([questions[i] for i in [1,2,3,4]], vi_mode=True)  # set up to select arbitrary subset of questions
                fake_parser = ArgumentParser()
                # copy t4a_attack_eval_command.py's register_subcommand behavior
                shared.add_arguments_model(fake_parser)
                shared.add_arguments_dataset(fake_parser, default_split="test")
                shared.add_arguments_attack(fake_parser)
                fake_parser.set_defaults(attack=True)
                fake_args = fake_parser.parse_args([f"--{k}={v}" for k, v in prompt_args.items() if k != 'confirmation']) # format for parser
                T4A_AttackEvalCommand().run(args=fake_args)
            elif prompt_action["action"] == "Train a model with train":
                prompt_args = prompt([questions[i] for i in [1,2,4]], vi_mode=True) # no need to ask q3, attack recipe. Set up to select arbitrary subset of questions
                fake_parser = ArgumentParser()
                # copy t4a_train_command.py's register_subcommand behavior
                shared.add_arguments_model(fake_parser)
                shared.add_arguments_dataset(fake_parser, default_split="train")
                shared.add_arguments_train(fake_parser, default_split_eval="test")
                fake_parser.set_defaults(attack=False)
                fake_args = fake_parser.parse_args([f"--{k}={v}" for k, v in prompt_args.items() if k != 'confirmation']) # format for parser
                T4A_TrainCommand().run(args=fake_args)
            elif prompt_action["action"] == "Adversarial training with attack_train":
                prompt_args = prompt([questions[i] for i in [1,2,3,4]], vi_mode=True) # set up to select arbitrary subset of questions
                fake_parser = ArgumentParser()
                # copy t4a_attack_train_command.py's register_subcommand behavior
                shared.add_arguments_model(fake_parser)
                shared.add_arguments_dataset(fake_parser, default_split="train")
                shared.add_arguments_attack(fake_parser)
                shared.add_arguments_train(fake_parser)
                fake_parser.set_defaults(attack=True)
                fake_args = fake_parser.parse_args([f"--{k}={v}" for k, v in prompt_args.items() if k != 'confirmation']) # format for parser
                T4A_AttackTrainCommand().run(args=fake_args)
        except InvalidArgument:
            print("No available choices")

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser_interactive = main_parser.add_parser(
            "interactive",
            help="start the interactive CLI to run a T4A command",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser_interactive.set_defaults(func=InteractiveCliCommand())
