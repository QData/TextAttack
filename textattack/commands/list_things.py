"""

TextAttack list Command
=====================================
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import textattack
from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import (
    ATTACK_RECIPE_NAMES,
    BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
    CONSTRAINT_CLASS_NAMES,
    GOAL_FUNCTION_CLASS_NAMES,
    HUGGINGFACE_DATASET_BY_MODEL,
    SEARCH_METHOD_CLASS_NAMES,
    TEXTATTACK_DATASET_BY_MODEL,
    WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
)
from textattack.commands.augment import AUGMENTATION_RECIPE_NAMES


def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class ListThingsCommand(TextAttackCommand):
    """The list module:

    List default things in textattack.
    """

    def _list(self, list_of_things, plain=False):
        """Prints a list or dict of things."""
        if isinstance(list_of_things, list):
            list_of_things = sorted(list_of_things)
            for thing in list_of_things:
                if plain:
                    print(thing)
                else:
                    print(_cb(thing))
        elif isinstance(list_of_things, dict):
            for thing in sorted(list_of_things.keys()):
                thing_long_description = list_of_things[thing]
                if plain:
                    thing_key = thing
                else:
                    thing_key = _cb(thing)
                print(f"{thing_key} ({thing_long_description})")
        else:
            raise TypeError(f"Cannot print list of type {type(list_of_things)}")

    @staticmethod
    def things():
        list_dict = {}
        list_dict["models"] = list(HUGGINGFACE_DATASET_BY_MODEL.keys()) + list(
            TEXTATTACK_DATASET_BY_MODEL.keys()
        )
        list_dict["search-methods"] = SEARCH_METHOD_CLASS_NAMES
        list_dict["transformations"] = {
            **BLACK_BOX_TRANSFORMATION_CLASS_NAMES,
            **WHITE_BOX_TRANSFORMATION_CLASS_NAMES,
        }
        list_dict["constraints"] = CONSTRAINT_CLASS_NAMES
        list_dict["goal-functions"] = GOAL_FUNCTION_CLASS_NAMES
        list_dict["attack-recipes"] = ATTACK_RECIPE_NAMES
        list_dict["augmentation-recipes"] = AUGMENTATION_RECIPE_NAMES
        return list_dict

    def run(self, args):
        try:
            list_of_things = ListThingsCommand.things()[args.feature]
        except KeyError:
            raise ValueError(f"Unknown list key {args.thing}")
        self._list(list_of_things, plain=args.plain)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "list",
            help="list features in TextAttack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "feature", help="the feature to list", choices=ListThingsCommand.things()
        )
        parser.add_argument(
            "--plain",
            help="print output without color",
            default=False,
            action="store_true",
        )
        parser.set_defaults(func=ListThingsCommand())
