from data_processor import DataProcessor
from surprise import NormalPredictor
from matrix_factorization_algo import MatrixFactorizationAlgo
# from Evaluator import Evaluator
#load dataset
data_processor = DataProcessor()
data_processor.SetDataPath('./ml-latest-small/')
ratings = DataProcessor.LoadRating(data_processor)
#construct evaluator

#call MFalgo generate algo
#call KNNalgo generate algo

#use random as our basline here
Random = NormalPredictor()
# evaluator.AddAlgorithm(Random, "Random")
mf_algo = MatrixFactorizationAlgo()
mf_algo_dict = mf_algo.generate_algorithms(ratings)
# for key in mf_algo_dict:
#     evaluator.AddAlgorithm(mf_algo_dict[key], key)
# evaluate
