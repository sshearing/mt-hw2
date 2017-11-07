from Decoder import Decoder
from collections import namedtuple

class MonotoneDecoder(Decoder):

    def __init__(self, tm, lm, maxsize):
        self.tm = tm
        self.lm = lm
        self.ms = maxsize
        self.stacks = {}
        self.hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")

    def initialize(self, sentence):

        initial_hypothesis = self.hypothesis(0.0, self.lm.begin(), None, None)
        self.stacks = [{} for _ in sentence] + [{}]
        self.stacks[0][self.lm.begin()] = initial_hypothesis

    def update(self, j, h, phrase, ls):

        # get new language model state and logprob
        logprob = h.logprob + phrase.logprob
        lm_state = h.lm_state
        for word in phrase.english.split():
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            logprob += word_logprob
        logprob += self.lm.end(lm_state) if j == ls else 0.0

        # create new hypothesis with backpointer
        new_hypothesis = self.hypothesis(logprob, lm_state, h, phrase)
        if lm_state not in self.stacks[j] or self.stacks[j][lm_state].logprob < logprob: # second is recombination
            self.stacks[j][lm_state] = new_hypothesis
        
    def decode(self, sentence):

        self.initialize(sentence)
        for i, stack in enumerate(self.stacks[:-1]):
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:self.ms]: # prune
                for j in xrange(i+1,len(sentence)+1):
                    if sentence[i:j] in self.tm:
                        for phrase in self.tm[sentence[i:j]]:
                            self.update(j, h, phrase, len(sentence))

        return max(self.stacks[-1].itervalues(), key=lambda h: h.logprob)
        
            
        
    
