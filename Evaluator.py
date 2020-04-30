from EvaluationDataSet import EvaluationData
from recommender_metrics import recommender_metrics

class Evaluator:
    algos = []

    def __init__(self,dataset,rank):
        self.dataset = EvaluationData(dataset, rank)#a

    def Add_Algo(self,algorithm):
        alg = recommender_metrics()
        self.algos.append(alg)

    def print(self,TopN):




