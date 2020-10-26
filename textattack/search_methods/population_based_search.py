"""
Population based Search
==========================
"""

from abc import ABC, abstractmethod

from textattack.search_methods import SearchMethod


class PopulationBasedSearch(SearchMethod, ABC):
    """Abstract base class for population-based search methods.

    Examples include: genetic algorithm, particle swarm optimization
    """

    def _check_constraints(self, transformed_text, current_text, original_text):
        """Check if `transformted_text` still passes the constraints with
        respect to `current_text` and `original_text`.

        This method is required because of a lot population-based methods does their own transformations apart from
        the actual `transformation`. Examples include `crossover` from `GeneticAlgorithm` and `move` from `ParticleSwarmOptimization`.
        Args:
            transformed_text (AttackedText): Resulting text after transformation
            current_text (AttackedText): Recent text from which `transformed_text` was produced from.
            original_text (AttackedText): Original text
        Returns
            `True` if constraints satisfied and `False` if otherwise.
        """
        filtered = self.filter_transformations(
            [transformed_text], current_text, original_text=original_text
        )
        return True if filtered else False

    @abstractmethod
    def _perturb(self, pop_member, original_result, **kwargs):
        """Perturb `pop_member` in-place.

        Must be overridden by specific population-based method
        Args:
            pop_member (PopulationMember): Population member to perturb\
            original_result (GoalFunctionResult): Result for original text. Often needed for constraint checking.
        Returns
            `True` if perturbation occured. `False` if not.
        """
        raise NotImplementedError()

    @abstractmethod
    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        raise NotImplementedError


class PopulationMember:
    """Represent a single member of population."""

    def __init__(self, attacked_text, result=None, attributes={}, **kwargs):
        self.attacked_text = attacked_text
        self.result = result
        self.attributes = attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def score(self):
        if not self.result:
            raise ValueError(
                "Result must be obtained for PopulationMember to get its score."
            )
        return self.result.score

    @property
    def words(self):
        return self.attacked_text.words

    @property
    def num_words(self):
        return self.attacked_text.num_words
