from abc import ABC, abstractmethod


class Coverage(ABC):
    """``Coverage`` class measures how well a given test dataset tests the
    given model.

    This is an abstract base class for other ``Coverage`` classes.
    """


class ExtrinsicCoverage(Coverage):
    """Represents coverage methods that do not access the model that is subject
    of testing to measure the quality of test set."""

    @abstractmethod
    def __call__(self, testset):
        raise NotImplementedError()


class IntrinsicCoverage(Coverage):
    """Represents coverage methods that do access the model that is subject of
    testing to measure the quality of test set."""

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def __call__(self, testset):
        raise NotImplementedError()
