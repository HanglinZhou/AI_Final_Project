
from Evaluator import Evaluator
from surprise import NormalPredictor
from matrix_factorization_algo import MatrixFactorizationAlgo
from DataHandler import DataHandler

from knn_algo import knn
from hybrid_algo_weighted import HybridAlgoWeighted

# from Evaluator import Evaluator
#load dataset
dataprocessor = DataHandler()
evaluationData = dataprocessor.getEvaluation()
rankings = dataprocessor.getRank()
evaluator = Evaluator()

#construct evaluator
evaluator = Evaluator()

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


#use random as our basline here
for key in mf_algo_dict:
    evaluator.Add_Algo(mf_algo_dict[key], key)

# hybrid_weighted_algorithms = {'SVD_tuned' : mf_algo_dict['SVD_tuned'], 'NMF' : mf_algo_dict['NMF']}
# hybrid_weighted_weights = {'SVD_tuned' : 0.7, 'NMF' :0.3}
# hybrid_weighted = HybridAlgoWeighted(hybrid_weighted_algorithms, hybrid_weighted_weights)
# evaluator.Add_Algo(hybrid_weighted, "Weighted Hybrid")

# evaluate
evaluator.print(True)
