class CoverageStacks():

    def __init__(self, sentence_length, threshold):
        self.t = threshold  
        self.stacks = [{} for _ in range(sentence_length)] + [{}]

    # insert a hypothesis into correct stack given key
    def insert(self, key, hypothesis):
        
        if tuple(key) not in self.stacks[hypothesis.covered]:
            self.stacks[hypothesis.covered][tuple(key)] = KeyStack(key, self.t)
        self.stacks[hypothesis.covered][tuple(key)].insert(hypothesis)

    # get a specific stack
    def getstack(self, key):
        index = len([i for i in key if i == 1])
        return self.stacks[index][tuple(key)]

    # return an iterable list of stacks
    def generator(self):
        for i in range(len(self.stacks) - 1):
            for stack in self.stacks[i].itervalues():
                yield stack

class KeyStack():
     
    def __init__(self, coverage, threshold):
        self.key = coverage
        self.alpha = threshold
        self.best = float('-inf')
        self.stack = {}

    # insert a hypothesis into the stack
    def insert(self, hypothesis):

        # if logprob is worse than threshold, don't add it in.
        if hypothesis.logprob > self.best:
            self.best = hypothesis.logprob
        elif hypothesis.logprob < self.best * self.alpha:
            return

        # add the hypothesis to the stack, organized by language model state
        if hypothesis.lm_state not in self.stack:
            self.stack[hypothesis.lm_state] = hypothesis

        # if hypothesis with same language modeel state exists, do recombination
        elif self.stack[hypothesis.lm_state].logprob < hypothesis.logprob:
            self.stack[hypothesis.lm_state] = hypothesis

    # get an iterator of the hypotheses in the stack
    def itervalues(self):
        return self.stack.itervalues()
