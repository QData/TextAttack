from abc import ABC, abstractmethod

from textattack.shared.utils import default_class_repr


class SearchMethod(ABC):
    """This is an abstract class that contains main helper functionality for
    search methods.

    A search method is a strategy for applying transformations until the
    goal is met or the search is exhausted.
    """

    def __call__(self, initial_result):
        """Ensures access to necessary functions, then calls
        ``_perform_search``"""
        if not hasattr(self, "get_transformations"):
            raise AttributeError(
                "Search Method must have access to get_transformations method"
            )
        if not hasattr(self, "get_goal_results"):
            raise AttributeError(
                "Search Method must have access to get_goal_results method"
            )
        if not hasattr(self, "filter_transformations"):
            raise AttributeError(
                "Search Method must have access to filter_transformations method"
            )
        return self._perform_search(initial_result)

    @abstractmethod
    def _perform_search(self, initial_result):
        """Perturbs `attacked_text` from ``initial_result`` until goal is
        reached or search is exhausted.

        Must be overridden by specific search methods.
        """
        raise NotImplementedError()

    def check_transformation_compatibility(self, transformation):
        """Determines whether this search method is compatible with
        ``transformation``."""
        return True

    def extra_repr_keys(self):
        return []

    __repr__ = __str__ = default_class_repr
