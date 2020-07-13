from abc import ABC, abstractmethod


class TextAttackCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
