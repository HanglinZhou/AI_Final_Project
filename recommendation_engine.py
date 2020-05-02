
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
# evaluator = Evaluator()

#call MFalgo generate algo
#call KNNalgo generate algo
knngenerator = knn()
knn_algo = knngenerator.untuned_knn_algo()

# tune knn algo
tuned_knn_algo = {}
best_k = {}
for key in knn_algo:
    evaluator.Add_Algo(knn_algo[key],key)

for key in knn_algo:
    print(type(key))
    tuned_knn_algo[key] = knngenerator.analyze_knn_model(evaluationData, key)

print("tune added")

for key in tuned_knn_algo:
    evaluator.Add_Algo(tuned_knn_algo[key],key)



#use random as our basline here
Random = NormalPredictor()
evaluator.Add_Algo(Random, "Random")

# adding MF algos
mf_algo = MatrixFactorizationAlgo()
mf_algo_dict = mf_algo.generate_algorithms(evaluationData)
for key in mf_algo_dict:
    evaluator.Add_Algo(mf_algo_dict[key], key)




#use random as our basline here
# for key in mf_algo_dict:
#     evaluator.Add_Algo(mf_algo_dict[key], key)
# mf_algo_dict = mf_algo.generate_algorithms(evaluationData)
# for key in mf_algo_dict:
#     evaluator.Add_Algo(mf_algo_dict[key], key)


# adding knn algos
# knn_algo = knn.untuned_knn_algo()
#
# # tune knn algo
# tuned_knn_algo = {}
# best_k = {}
# for key in knn_algo:
#     best_k[key], tuned_knn_algo[key] = knn.analyze_knn_model(evaluationData,key)

# adding hybrid algos
# hybrid_weighted_algorithms = {'SVD' : mf_algo_dict['SVD'], 'NMF' : mf_algo_dict['NMF']}
# hybrid_weighted_weights = {'SVD' : 0.7, 'NMF' :0.3}
# hybrid_weighted = HybridAlgoWeighted(hybrid_weighted_algorithms, hybrid_weighted_weights)
# evaluator.Add_Algo(hybrid_weighted, "Weighted Hybrid")


# hybrid_weighted_algorithms = {'SVD_tuned' : mf_algo_dict['SVD_tuned'], 'NMF' : mf_algo_dict['NMF']}
# hybrid_weighted_weights = {'SVD_tuned' : 0.7, 'NMF' :0.3}
# hybrid_weighted = HybridAlgoWeighted(hybrid_weighted_algorithms, hybrid_weighted_weights)
# evaluator.Add_Algo(hybrid_weighted, "Weighted Hybrid")

# evaluate
evaluator.print(True)
