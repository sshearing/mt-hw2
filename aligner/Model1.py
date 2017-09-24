import math
from Aligner import Aligner
from collections import defaultdict

class Model1(Aligner):

    def __init__(self):
        self.t = defaultdict(int)
        
    # train an IBM Model1 Word Aligner
    def train(self, bitext):

        # create a mapping of possible (e, f) pairs
        pairs = defaultdict(set)
        for (f, e) in bitext:
            for e_j in e:
                for f_i in f:
                    pairs[e_j].add(f_i)

        # initialize t uniformly across possible pairs
        for e in pairs:
            for f in pairs[e]:
                self.t[(e, f)] = 1.0 / float(len(pairs[e]))

        old = float("inf")
        convergent = False
        n = 0
        while not convergent:

            # initialize
            count = defaultdict(float)
            total = defaultdict(float)

            for (f, e) in bitext:

                # compute normalization
                s_total = defaultdict(float)
                for e_i in e:
                    for f_j in f:
                        s_total[e_i] += self.t[(e_i, f_j)]

                # collect counts
                for e_i in e:
                    for f_j in f:
                        count[(e_i, f_j)] += self.t[(e_i, f_j)] / s_total[e_i]
                        total[f_j] += self.t[(e_i, f_j)] / s_total[e_i]

            # estimate probabilities
            for e in pairs:
                for f in pairs[e]:
                    self.t[(e, f)] = count[(e, f)] / total[f]
        
            # calculate perplexity
            # perp = 0.0
            # for (f, e) in bitext:
             #   perp += -1.0 * math.log(1.0 / (len(f) + 1) ** len(e) * reduce(lambda x, y: x * y, [sum([self.t[(e_j, f_i)] for f_i in f]) for e_j in e]) / math.log(2))

            # check to see if we have converged
            # if perp + 0.1 < old:
             #   old = perp
            #else:
            if n == 20:
                convergent = True
            n += 1

    # given a sentence, output most probable alignment
    def align(self, (f, e)):
        alignment = []
        for (j, e_j) in enumerate(e):
            best_prob = 0
            best_i = 0
            for (i, f_i) in enumerate(f):
                if self.t[(e_j, f_i)] > best_prob:
                    best_prob = self.t[(e_j, f_i)]
                    best_i = i
            alignment.append((best_i, j))
                
            # a.append((j, max(enumerate([self.t[(e_j, f_i)] for f_i in f]), key=lambda item:item[1])[0]))
        return alignment
