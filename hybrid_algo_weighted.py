"""
A hybrid algorithm that incorporates input algorithms and generates a hybrid algorithm.
Different recommendation algorithms are combined based on given weights (all weights are sum up to 1).
"""
from surprise import AlgoBase

class HybridAlgoWeighted(AlgoBase):
    # store input algorithms as a dictionary: key: name of algo, value: algo object
    algorithms = {}

    # store input weights as a dictionary: key: name of algo, value: weight
    weights = {}

    weighted_estimate = 0

    # constructor: initialize our algorithms and weights
    def __init__(self, algorithms, weights):
        AlgoBase.__init__(self)
        if (sum(weights.values()) - 1) != 0:
            raise Exception("Attention, sum of weights need to be 1")
       # print("in constructor, algos:", type(algorithms))
        self.algorithms = algorithms
        self.weights = weights

    # run fit() for all algo
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        for algoName, algo in self.algorithms.items():
            print(type(algo))
            algo.fit(trainset)
        return self

    # derived from AlgoBase: u as uid: (Raw) id of the user; i as iid: (Raw) id of the item.
    def estimate(self, u, i):
        #print('Hybrid Algo included (algo with weight): ')
        for algo in self.algorithms:
            # sum of (each algo's estimate * its weight) = weighted_estimate
            #print('', algo, ' with ', self.weights[algo])
            self.weighted_estimate += self.algorithms[algo].estimate(u, i) * self.weights[algo]

        #print('Hybrid Algo Weighted estimate: ', self.weighted_estimate)
        return self.weighted_estimate
