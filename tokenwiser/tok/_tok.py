from abc import ABC, abstractmethod


class Tok(ABC):
    @abstractmethod
    def __call__(self, x):
        pass
