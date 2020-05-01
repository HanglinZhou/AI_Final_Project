
from Evaluator import Evaluator
from surprise import NormalPredictor
from matrix_factorization_algo import MatrixFactorizationAlgo
from EvaluationDataSet import EvaluationDataSet

from knn_algo import knn
# from Evaluator import Evaluator
#load dataset
dataprocessor = EvaluationDataSet()
evaluationData = dataprocessor.getEvaluation()
rankings = dataprocessor.getRank()
evaluator = Evaluator()

#construct evaluator

#call MFalgo generate algo
#call KNNalgo generate algo
# knn_algo = knn.KnnAlgo()

#use random as our basline here
Random = NormalPredictor()
evaluator.Add_Algo(Random, "Random")
mf_algo = MatrixFactorizationAlgo()
mf_algo_dict = mf_algo.generate_algorithms(evaluationData)
for key in mf_algo_dict:
    evaluator.Add_Algo(mf_algo_dict[key], key)
    #why svd print twice?? the second time no
evaluator.print(True)


