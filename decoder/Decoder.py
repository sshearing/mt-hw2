from abc import ABCMeta, abstractmethod

class Decoder:
    __metaclass__ = ABCMeta

    @abstractmethod
    def decode(self, sentence): pass

    def extract_english(self, h):
        return "" if h.predecessor is None else "%s%s " % (self.extract_english(h.predecessor), h.phrase.english)

    def extract_tm_logprob(self, h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + self.extract_tm_logprob(h.predecessor)
                
                



