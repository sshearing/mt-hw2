from operator import itemgetter
from itertools import groupby
from Decoder import Decoder
from collections import namedtuple
from CoverageStack import CoverageStacks, KeyStack

class CoverageStackDecoder(Decoder):

    def __init__(self, tm, lm, maxsize, reorderlimit, threshold):
        self.tm = tm
        self.lm = lm
        self.ms = maxsize
        self.rl = reorderlimit
        self.th = threshold
        self.stacks = {}
        self.hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, covered")

    def initialize(self, sentence):

        initial_hypothesis = self.hypothesis(0.0, self.lm.begin(), None, None, 0)
        self.stacks = CoverageStacks(len(sentence), self.th)
        self.stacks.insert([0] * len(sentence), initial_hypothesis)

    def update(self, j, k, h, phrase, key, ls):

        # get new language model state and logprob
        logprob = h.logprob + phrase.logprob
        lm_state = h.lm_state
        for word in phrase.english.split():
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            logprob += word_logprob

        # create new covered list
        covered = list(key)
        for i in xrange(j, k):
            covered[i] = 1

        # if we have a full translation hypothesis, add in end state to logprob.
        if len([i for i in covered if i == 1]) == ls:
            logprob += self.lm.end(lm_state)

        # create new hypothesis with backpointer
        new_hypothesis = self.hypothesis(logprob, lm_state, h, phrase, h.covered + k - j)

        self.stacks.insert(covered, new_hypothesis)
        
        
    def decode(self, sentence):
        
        self.initialize(sentence)
        for stack in self.stacks.generator():

            # build valid phrase indexes, split into ranges for easy iteration
            uncovered = [index for index, value in enumerate(stack.key) if value == 0]
            ranges = [map(itemgetter(1), g) for k, g in groupby(enumerate(uncovered), lambda (i, x):i-x)]
            
            # determine maximum starting phrase position given reordering limit
            limit = uncovered[min(len(uncovered) - 1, self.rl)]
            
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:self.ms]: # prune

                # for every possible phrase in the valid indexes, build a hypothesis
                for r in ranges:
                    for j in xrange(r[0], min(r[-1], limit)+1):
                        for k in xrange(j+1,r[-1]+2):
                            if sentence[j:k] in self.tm:
                                for phrase in self.tm[sentence[j:k]]:
                                    self.update(j, k, h, phrase, stack.key, len(sentence))

        return max(self.stacks.getstack([1] * len(sentence)).itervalues(), key=lambda h: h.logprob)
        
            
        
    
