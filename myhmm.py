import json
import os
import sys


class MyHmm(object):  # base class for different HMM models
    def __init__(self, model_name):
        # model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
        # a model can be initialized to random parameters using a json file that has a random params model
        if model_name is None:
            print("Fatal Error: You should provide the model file name")
            sys.exit()
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"]
        self.states = self.A.keys()  # get the list of states
        self.N = len(self.states)  # number of states of the model
        self.B = self.model["B"]
        self.symbols = list(self.B.values())[
            0].keys()  # get the list of symbols, assume that all symbols are listed in the B matrix
        self.M = len(self.symbols)  # number of states of the model
        self.pi = self.model["pi"]
        return

    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T - 1][y] = 1  # self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T - 1)):
            for y in self.states:
                self.bwk[t][y] = sum(
                    (self.bwk[t + 1][y1] * self.A[y][y1] * self.B[y1][obs[t + 1]]) for y1 in self.states)
        prob = sum((self.pi[y] * self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        return prob

    def forward(self, obs):
        self.fwd = [{}]
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob

    def viterbi(self, obs):
        vit = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]

        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t - 1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
                # Don't need to remember the old paths
            path = newpath
        n = 0  # if only one element is observed max is sought in the initialization values
        if len(obs) != 1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward(self, obs):  # returns model given the initial model and observations
        gamma = [{} for t in
                 range(len(obs))]  # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(
            len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                # compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)])  #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.symbols:  # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k:
                        val += gamma[t][y]
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
        return