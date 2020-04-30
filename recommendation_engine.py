from data_processor import DataProcessor
from surprise import NormalPredictor

#load dataset
data_processor = DataProcessor()
data_processor.SetDataPath('./ml-latest-small/')
ratings = DataProcessor.LoadRating(data_processor)

#call MFalgo generate algo
#call KNNalgo generate algo

#use random as our basline here
Random = NormalPredictor()
# evaluator.AddAlgorithm(Random, "Random")


# evaluate

#