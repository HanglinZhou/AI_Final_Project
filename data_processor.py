#This class is used for read and load data.

from surprise import Dataset
from surprise import Reader

class DataProcessor:
    data_path = ''
    rating = data_path + 'ratings.csv'
    movies = data_path + 'movies.csv'

    def SetDataPath(self, data_path):
        self.data_path = data_path
    def LoadRating(self):
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        return Dataset.load_from_file(self.rating, reader=reader)