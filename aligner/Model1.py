import math
from Aligner import Aligner
from collections import defaultdict

class Model1(Aligner):

    def __init__(self, iterations):
        self.t = defaultdict(int)
        self.iterations = iterations
        
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

        for epoch in range(self.iterations):

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
                elif self.t[(e_j, f_i)] == best_prob and abs(j - i) < abs(j - best_i):
                    best_prob = self.t[(e_j, f_i)]
                    best_i = i
            alignment.append((best_i, j))
                
        return alignment
