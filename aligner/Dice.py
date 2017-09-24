from Aligner import Aligner
from collections import defaultdict

class Dice(Aligner):

    def __init__(self, threshold):
        self.dice = defaultdict(int)
        self.threshold = threshold

    def train(self, bitext):
        
        f_count = defaultdict(int)
        e_count = defaultdict(int)
        fe_count = defaultdict(int)
        
        for (n, (f, e)) in enumerate(bitext):
            for f_i in set(f):
                f_count[f_i] += 1
                for e_j in set(e):
                    fe_count[(f_i,e_j)] += 1
            for e_j in set(e):
                e_count[e_j] += 1

        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            self.dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])

    def align(self, (f, e)):
        alignment = []
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                if self.dice[(f_i, e_j)] >= self.threshold:
                    alignment.append((i, j))
        return alignment
