from data_processor import DataProcessor

#load dataset
data_processor = DataProcessor()
data_processor.SetDataPath('./ml-latest-small/')
ratings = DataProcessor.LoadRating(data_processor)