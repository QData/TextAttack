from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import *
from textattack.commands.attack.attack_args_helpers import *


def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class EvalModelCommand(TextAttackCommand):
    """
    The TextAttack model benchmarking module:
    
        A command line parser to evaluatate a model from user specifications.
    """

    def get_num_successes(self, model, ids, true_labels):
        with torch.no_grad():
            preds = textattack.shared.utils.model_predict(model, ids)
        true_labels = torch.tensor(true_labels).to(textattack.shared.utils.device)
        guess_labels = preds.argmax(dim=1)
        successes = (guess_labels == true_labels).sum().item()
        return successes, true_labels, guess_labels

    def test_model_on_dataset(self, args):
        model = parse_model_from_args(args)
        dataset = parse_dataset_from_args(args)
        succ = 0
        fail = 0
        batch_ids = []
        batch_labels = []
        all_true_labels = []
        all_guess_labels = []
        for i, (text_input, label) in enumerate(dataset):
            if i >= args.num_examples:
                break
            attacked_text = textattack.shared.AttackedText(text_input)
            ids = model.tokenizer.encode(attacked_text.tokenizer_input)
            batch_ids.append(ids)
            batch_labels.append(label)
            if len(batch_ids) == args.batch_size:
                batch_succ, true_labels, guess_labels = self.get_num_successes(
                    model, batch_ids, batch_labels
                )
                batch_fail = args.batch_size - batch_succ
                succ += batch_succ
                fail += batch_fail
                batch_ids = []
                batch_labels = []
                all_true_labels.extend(true_labels.tolist())
                all_guess_labels.extend(guess_labels.tolist())
        if len(batch_ids) > 0:
            batch_succ, true_labels, guess_labels = self.get_num_successes(
                model, batch_ids, batch_labels
            )
            batch_fail = len(batch_ids) - batch_succ
            succ += batch_succ
            fail += batch_fail
            all_true_labels.extend(true_labels.tolist())
            all_guess_labels.extend(guess_labels.tolist())

        perc = float(succ) / (succ + fail) * 100.0
        perc = "{:.2f}%".format(perc)
        print(f"Successes {succ}/{succ+fail} ({_cb(perc)})")
        return perc

    def run(self, args):
        # Default to 'all' if no model chosen.
        if not (args.model or args.model_from_huggingface or args.model_from_file):
            for model_name in list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
                TEXTATTACK_DATASET_BY_MODEL.keys()
            ):
                args.model = model_name
                self.test_model_on_dataset(args)
        else:
            self.test_model_on_dataset(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "eval",
            help="evaluate a model with TextAttack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )

        add_model_args(parser)
        add_dataset_args(parser)

        parser.add_argument(
            "--batch-size",
            type=int,
            default=256,
            help="Batch size for model inference.",
        )
        parser.set_defaults(func=EvalModelCommand())
