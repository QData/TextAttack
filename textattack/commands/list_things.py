from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack.commands import TextAttackCommand
from textattack.commands.attack.attack_args import *
from textattack.commands.augment import AUGMENTATION_RECIPE_NAMES


def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class ListThingsCommand(TextAttackCommand):
    """
    The list module:
    
        List default things in textattack.
    """

    def _list(self, list_of_things):
        """ Prints a list or dict of things. """
        if isinstance(list_of_things, list):
            list_of_things = sorted(list_of_things)
            for thing in list_of_things:
                print(_cb(thing))
        elif isinstance(list_of_things, dict):
            for thing in sorted(list_of_things.keys()):
                thing_long_description = list_of_things[thing]
                print(f"{_cb(thing)} ({thing_long_description})")
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
            raise ValuError(f"Unknown list key {args.thing}")
        self._list(list_of_things)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "list",
            help="list features in TextAttack",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "feature", help=f"the feature to list", choices=ListThingsCommand.things()
        )
        parser.set_defaults(func=ListThingsCommand())
