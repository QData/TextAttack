from abc import ABC, abstractmethod
from argparse import ArgumentParser, HelpFormatter


class TextAttackCommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()


from . import textattack_cli
