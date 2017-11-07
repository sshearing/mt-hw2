from operator import itemgetter
from itertools import groupby
from Decoder import Decoder
from collections import namedtuple

class LimitedReorderDecoder(Decoder):

    def __init__(self, tm, lm, maxsize, reorderlimit, threshold):

        # model and option inputs
        self.tm = tm
        self.lm = lm
        self.ms = maxsize
        self.rl = reorderlimit
        self.th = threshold

        # initialize necessary data structures
        self.stacks = {}
        self.cost = {}
        self.best = []
        self.hypothesis = namedtuple("hypothesis", "logprob, cost, lm_state, predecessor, phrase, covered, num")

    def initialize(self, sentence):

        # calculate future cost estimates
        self.cost = {}
        for length in xrange(1, len(sentence)):
            for start in range(len(sentence) + 1 - length):
                end = start + length
                self.cost[(start, end)] = float('-inf')
                if sentence[start:end] in self.tm:
                    for phrase in self.tm[sentence[start:end]]:
                        option_cost = phrase.logprob
                        lm_state = ()
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = self.lm.score(lm_state, word)
                            option_cost += word_logprob
                        if option_cost > self.cost[(start, end)]:
                            self.cost[(start, end)] = option_cost 
                for i in xrange(start + 1, end):
                    if self.cost[(start, i)] + self.cost[(i,end)] > self.cost[(start, end)]:
                        self.cost[(start,end)] = self.cost[(start, i)] + self.cost[(i,end)]
        
        initial_hypothesis = self.hypothesis(0.0, self.cost[(0, len(sentence) - 1)], self.lm.begin(), None, None, [0] * len(sentence), 0)
        self.stacks = [{} for _ in sentence] + [{}]
        self.stacks[0][(self.lm.begin(), tuple([0] * len(sentence)))] = initial_hypothesis
        self.best = [float('-inf') for i in range(len(sentence) + 1)]
        
    def update(self, j, k, h, phrase, ls):

        # get new language model state and logprob
        logprob = h.logprob + phrase.logprob
        lm_state = h.lm_state
        for word in phrase.english.split():
            (lm_state, word_logprob) = self.lm.score(lm_state, word)
            logprob += word_logprob

        # create new covered list
        covered = list(h.covered)
        for i in xrange(j, k):
            covered[i] = 1

        # if we have a full translation hypothesis, add in end state to logprob.
        if len([i for i in covered if i == 1]) == ls:
            logprob += self.lm.end(lm_state)

        # estimate future cost
        uncovered = [index for index, value in enumerate(covered) if value == 0]
        spans = [map(itemgetter(1), g) for f, g in groupby(enumerate(uncovered), lambda (i, x):i-x)]
        future_cost = 0.0
        for span in spans:
            future_cost += self.cost[(span[0], span[-1] + 1)]

        # create new hypothesis with backpointer
        nh = self.hypothesis(logprob, future_cost, lm_state, h, phrase, covered, h.num + (k - j))

        covered = tuple(covered)

        # if hypothesis is below threshold, ignore it
        if logprob + future_cost > self.best[nh.num]:
            self.best[nh.num] = logprob + future_cost
        elif logprob + future_cost < self.best[nh.num] * self.th:
            return

        # add hypothesis to stack
        if (lm_state, covered) not in self.stacks[h.num + k - j]:
            self.stacks[nh.num][(lm_state, covered)] = nh

        # do recombination if necessary
        elif self.stacks[nh.num][(lm_state, covered)].logprob < logprob:
            self.stacks[nh.num][(lm_state, covered)] = nh
        
    def decode(self, sentence):

        self.initialize(sentence)
        for stack in self.stacks[:-1]:
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob - h.cost)[:self.ms]: # prune

                # build valid phrase indexes, split into ranges for easy iteration
                uncovered = [index for index, value in enumerate(h.covered) if value == 0]
                ranges = [map(itemgetter(1), g) for k, g in groupby(enumerate(uncovered), lambda (i, x):i-x)]

                # determine maximum starting phrase position given reordering limit
                limit = uncovered[min(len(uncovered) - 1, self.rl)]

                # for every possible phrase in the valid indexes, build a hypothesis
                for r in ranges:
                    for j in xrange(r[0], min(r[-1], limit)+1):
                        for k in xrange(j+1,r[-1]+2):
                            if sentence[j:k] in self.tm:
                                for phrase in self.tm[sentence[j:k]]:
                                    self.update(j, k, h, phrase, len(sentence))

        return max(self.stacks[-1].itervalues(), key=lambda h: h.logprob)
        
            
        
    
