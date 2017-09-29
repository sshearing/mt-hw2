import sys
import math
from Aligner import Aligner
from collections import defaultdict

class HMM(Aligner):

    def __init__(self, iterations):
        self.pt = defaultdict(int) # translation probabilities
        self.tt = defaultdict(lambda: 1.0) # transition probabilities
        self.iterations = iterations # number of EM iterations
        
    # Train HMM model
    def train(self, bitext):
        self.initialization(bitext)
        self.refinement(bitext)

    # Train transition probabilities
    def refinement(self, bitext):

        # initialize tt uniformly across sentence lengths
        for I in range(1, max([len(f) for (f, e) in bitext])):
            for j in range(I):
                for i in range(I):
                    self.tt[(i, j, I)] = 1.0 / I

        for epoch in range(self.iterations):

            # initialize parameters
            s = defaultdict(int)

            # collect parameters
            for (f, e) in bitext:
                a = self.align((f, e))
                for i in range(len(a) - 1):
                    s[abs(a[i+1][0] - a[i][0])] += 1.0 / len(e)
            
            # estimate transition probabilities
            for I in range(1, max([len(f) for (f, e) in bitext])):
                for j in range(I):
                    total = sum([s[k - j] for k in range(I)])
                    for i in range(I):
                        self.tt[(i, j, I)] = s[abs(i - j)] / total
        
    # train translation probabilities
    def initialization(self, bitext):

        # create a mapping of possible (e, f) pairs
        pairs = defaultdict(set)
        for (f, e) in bitext:
            for e_j in e:
                for f_i in f:
                    pairs[e_j].add(f_i)

        # initialize t uniformly across possible pairs
        for e in pairs:
            for f in pairs[e]:
                self.pt[(e, f)] = 1.0 / float(len(pairs[e]))

        for epoch in range(self.iterations):

            # initialize
            count = defaultdict(float)
            total = defaultdict(float)

            for (f, e) in bitext:

                # compute normalization
                s_total = defaultdict(float)
                for e_i in e:
                    for f_j in f:
                        s_total[e_i] += self.pt[(e_i, f_j)]

                # collect counts
                for e_i in e:
                    for f_j in f:
                        count[(e_i, f_j)] += self.pt[(e_i, f_j)] / s_total[e_i]
                        total[f_j] += self.pt[(e_i, f_j)] / s_total[e_i]

            # estimate probabilities
            for e in pairs:
                for f in pairs[e]:
                    self.pt[(e, f)] = count[(e, f)] / total[f]

    # return an alignment of (f, e) based on most likely approximation
    # dynamic programming formula:
    # Q(j, i) = p(e_j | f_i) max [p(i | i', I) * Q(j -1 , i')] 
    def align(self, (f, e)):
        
        # initialize Q and backpointers
        bp = {}
        Q = defaultdict(int)

        # calculate Q(0, i)
        for (i, f_i) in enumerate(f):
            Q[(0, i)] = self.pt[(e[0], f_i)] 

        # calculate Q(j, i) and backpointers
        for (j, e_j) in enumerate(e[1:], 1):
            for (i, f_i) in enumerate(f):
                best_prob = -1
                for k in range(len(f)):
                    Qprime = self.tt[(i, k, len(f))] * Q[(j - 1, k)]
                    if Qprime > best_prob:
                        Q[(j, i)] = self.pt[(e_j, f_i)] * Qprime
                        bp[(j, i)] = (j - 1, k)
                        best_prob = Qprime
                    elif Qprime == best_prob and abs(j - 1 - k) < abs(j - 1 - bp[(j, i)][1]):
                        bp[(j, i)] = (j - 1, k)

        # add last alignment
        alignment = []
        j, best = len(e) - 1, 0
        for i in range(len(f)):
            if Q[(j, i)] >= Q[(j, i)]:
                best = i
        alignment.append((i, j))

        # use backpointers to generate rest of alignment
        while not j == 0:
            j, i = bp[(j, i)]
            alignment.append((i, j))

        return alignment
