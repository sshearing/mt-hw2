import math
from Aligner import Aligner
from collections import defaultdict

class Model1(Aligner):

    def __init__(self):
        self.t = defaultdict(float)
        
    # train an IBM Model1 Word Aligner
    def train(self, bitext):

        # initialize t probabilities to uniform values
        e_words = set([e_word for (f, e) in bitext for e_word in e])
        f_words = set([f_word for (f, e) in bitext for f_word in f])
        self.t = defaultdict(lambda: float(len(e_words) * len(f_words)))

        old = float("inf")
        convergent = False
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
            for f in f_words:
                for e in e_words:
                    self.t[(e, f)] = count[(e, f)] / total[f]
        
            # calculate perplexity
            perp = 0.0
            for (f, e) in bitext:
                perp += -1.0 * math.log(1.0 / (len(f) + 1) ** len(e) * reduce(lambda x, y: x * y, [sum([self.t[(e_j, f_i)] for f_i in f]) for e_j in e]) / math.log(2))

            # check to see if we have converged
            if perp < old:
                old = perp
            else:
                convergent = True

    # given a sentence, output most probable alignment
    def align(self, (f, e)):
        a = []
        for (j, e_j) in enumerate(e):
            a.append((j, max(enumerate([self.t[(e_j, f_i)] for f_i in f]), key=lambda item:item[1])[0]))
        return a
