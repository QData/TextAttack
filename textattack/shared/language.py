from abc import ABC, abstractmethod

class Language(ABC):

    @abstractmethod
    def language_code(self):
        raise NotImplementedError