class algorithm_eval:

    # constructor
    # algorithm: the prediction algorihtm we use
    # name: algorithm name
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        return metrics

    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm
