from eukaryote import AttackArgs, Attacker
from eukaryote.constraints import PreTransformationConstraint
from eukaryote.constraints.pre_transformation import MaxModificationRate

from eukaryote.t4a.attack_eval_support.results import Results

default_perturbation_budget_class = MaxModificationRate


def run_attack(
    model_wrapper,
    attack_recipe,
    dataset,
    perturbation_budget=None,
    perturbation_budgets=None,
    perturbation_budget_class=default_perturbation_budget_class,
    attack_strength=None,
    attack_strengths=None,
    attack_args=None,
    attack_build_args=None,
):
    """Run an attack on a model, attack recipe, and dataset.

    Only one of the following be provided:
        - `perturbation_budget`
        - `perturbation_budgets`
        - `attack_strength`
        - `attack_strengths`

    Args:
        model_wrapper (eukaryote.models.wrappers.ModelWrapper):
            The model to attack as a eukaryote `ModelWrapper`.
        attack_recipe (eukaryote.attack_recipes.AttackRecipe):
            The attack recipe.
        dataset (eukaryote.datasets.Dataset):
            The dataset to attack.
        perturbation_budget (Optional[Union[dict, Any]]):
            When provided, adds a perturbation budget constraint defined by
            `perturbation_budget_class` with the given kwargs or single
            argument.
        perturbation_budgets (Optional[list[Union[dict, Any]]]):
            Broadcasted version of `perturbation_budget`.
        perturbation_budget_class (Union[
            eukaryote.constraints.PreTransformationConstraint,
            eukaryote.constraints.Constraint,
            ]):
            The class of the constraint which defines perturbation budget.
            Defaults to
            `eukaryote.constraints.pre_transformation.MaxModificationRate`.
        attack_strength (Optional[int]):
            When provided, constrains the attacker by an upper bound of
            iterations (also know as a query budget).
        attack_strengths (Optional[int]):
            Broadcasted version of `attack_strength`.
        attack_args (Optional[Union[dict, list[dict]]):
            Additional kwargs to pass to a `eukaryote.AttackArgs`. If a list
            is passed, then applies kwargs separately to each attack.
        attack_build_args (Optional[Union[dict, list[dict]]):
            Additional kwargs to pass to `attack_recipe.build(model_wrapper)`.
            If a list is passed, then applies kwargs separately to each attack.

    Returns:
        tats4aardvarks.attack_eval.Results
    """

    if perturbation_budgets is None and perturbation_budget is not None:
        perturbation_budgets = [perturbation_budget]
    if attack_strengths is None and attack_strength is not None:
        attack_strengths = [attack_strength]
    attacker = MultiAttacker(
        model_wrapper,
        attack_recipe,
        perturbation_budgets,
        perturbation_budget_class,
        attack_strengths,
        attack_args,
        attack_build_args,
    )
    return attacker.attack_dataset(dataset)


class MultiAttacker:
    """Contain a `eukaryote.Attacker` in order to optionally broadcast extra
    perturbation budget or attack strength constraints.

    This abstraction allows for potential optimization (if implemented) of
    attacking when done over the same attack with varying extra constraints.
    """

    def __init__(
        self,
        model_wrapper,
        attack_recipe,
        perturbation_budgets=None,
        perturbation_budget_class=default_perturbation_budget_class,
        attack_strengths=None,
        attack_args=None,
        attack_build_args=None,
    ):
        self.model_wrapper = model_wrapper

        if not ((perturbation_budgets is None) or (attack_strengths is None)):
            raise ValueError(
                "Only one of perturbation_budgets or attack_strengths can be passed"
            )

        # Construct attack build args
        if perturbation_budgets is not None:
            num_attacks = len(perturbation_budgets)
        elif attack_strengths is not None:
            num_attacks = len(attack_strengths)
        else:
            num_attacks = None
        if num_attacks is not None:
            if isinstance(attack_build_args, list):
                if len(attack_build_args) != num_attacks:
                    raise ValueError("Unequal number of attack build args and attacks")
            else:
                if attack_build_args is None:
                    attack_build_args = {}
                attack_build_args = [attack_build_args] * num_attacks
        elif attack_build_args is None:
            attack_build_args = {}

        # Construct list of attacks (and default labels)
        self.attacks = []
        self.labels = []
        if perturbation_budgets is not None:
            for pb, attack_build_kwargs in zip(perturbation_budgets, attack_build_args):
                attack = attack_recipe.build(model_wrapper, **attack_build_kwargs)
                if pb is not None:
                    if isinstance(pb, dict):
                        pb_constraint = perturbation_budget_class(**pb)
                    else:
                        pb_constraint = perturbation_budget_class(pb)
                    if isinstance(pb_constraint, PreTransformationConstraint):
                        attack.pre_transformation_constraints.append(pb_constraint)
                    else:
                        attack.constraints.append(pb_constraint)
                self.attacks.append(attack)
                self.labels.append(f"Attack (pb.={pb})")
        elif attack_strengths is not None:
            for qb, attack_build_kwargs in zip(attack_strengths, attack_build_args):
                attack = attack_recipe.build(model_wrapper, **attack_build_kwargs)
                if qb is not None:
                    attack.goal_function.query_budget = qb
                self.attacks.append(attack)
                self.labels.append(f"Attack (as.={qb})")
        else:
            self.attacks.append(attack_recipe.build(model_wrapper, **attack_build_args))
            self.labels.append("Attack")

        # Construct attack args
        if isinstance(attack_args, list):
            if len(attack_args) != len(self.attacks):
                raise ValueError("Unequal number of attack args and attacks")
            self.attack_args = attack_args
        else:
            if attack_args is None:
                attack_args = {}
            self.attack_args = [attack_args] * len(self.attacks)

    def attack_dataset(self, dataset):
        attack_results = []
        for attack, attack_args in zip(self.attacks, self.attack_args):
            if "num_examples" not in attack_args:
                attack_args["num_examples"] = len(dataset)
            attacker = Attacker(attack, dataset, AttackArgs(**attack_args))
            attack_results.append(attacker.attack_dataset())
        return Results.from_attack_results(attack_results, self.labels)
