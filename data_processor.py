#This class is used for read and load data.
import csv
from collections import defaultdict

from surprise import Dataset
from surprise import Reader

class DataProcessor:
    data_path = ''
    rating = ''
    movies = ''

    def SetDataPath(self, data_path):
        self.data_path = data_path
        self.rating = data_path + 'ratings.csv'
        self.movies = data_path + 'movies.csv'
    def LoadRating(self):
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        return Dataset.load_from_file(self.rating, reader=reader)

    def loadPopularityData(self):
        #similart to getOrDefault in Java
        ratingTimes = defaultdict(int)
        rankings = defaultdict(int)

        with open(self.rating, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                movieId = int(row[1])
                ratingTimes[movieId]+=1
        rank = 1

        for movieID, count in sorted(ratingTimes.items(),key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank +=1
        return rankings



