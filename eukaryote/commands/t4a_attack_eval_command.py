import eukaryote.t4a.shared as shared
import eukaryote.t4a.attack_eval_support.attack as attack_eval

from eukaryote.commands import TextAttackCommand


class T4A_AttackEvalCommand(TextAttackCommand):
    def run(self, args):
        """Runs an attack with a perturbation budget and strength, and then prints out a table"""
        model_wrapper = shared.load_model_wrapper(args)
        dataset = shared.load_dataset(args)
        attack_obj = shared.load_attack(args)

        results = attack_eval.run_attack(
            model_wrapper,
            attack_obj["attack_recipe"],
            dataset,
            perturbation_budgets=attack_obj.get("perturbation_budget"),
            perturbation_budget_class=attack_obj.get("perturbation_budget_class"),
            attack_strengths=attack_obj.get("attack_strength"),
            attack_args=attack_obj.get("attack_args"),
        )

        print(results.create_table_results())

    @staticmethod
    def register_subcommand(subparsers):
        parser = subparsers.add_parser(
            "t4a_attack_eval",
            description="Attack and evaluate a model",
        )
        shared.add_arguments_model(parser)
        shared.add_arguments_dataset(parser, default_split="test")
        shared.add_arguments_attack(parser)
        parser.set_defaults(func=T4A_AttackEvalCommand())
