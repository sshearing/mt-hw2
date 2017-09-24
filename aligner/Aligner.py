from abc import ABCMeta, abstractmethod

class Aligner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, bitext): pass

    @abstractmethod
    def align(self, sentence): pass
