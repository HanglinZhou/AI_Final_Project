#import self as self
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline
from surprise import Trainset
import csv
from collections import defaultdict

from surprise import Dataset
from surprise import Reader


class DataHandler:

    # rating = './ml-latest-small/ratings.csv'
    # movies = './ml-latest-small/movies.csv'

    # for testing purpose
    rating = './test-data/ratings.csv'
    movies = './test-data/movies.csv'
    def LoadRating(self):
        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        return Dataset.load_from_file(self.rating, reader=reader)

    def loadPopularityData(self):
        # similart to getOrDefault in Java
        ratingTimes = defaultdict(int)
        rankings = defaultdict(int)

        with open(self.rating, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                movieId = int(row[1])
                ratingTimes[movieId] += 1
        rank = 1

        for movieID, count in sorted(ratingTimes.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings

    # def __init__(self):
    #     self.evaluataiondata=self.LoadRating()
    #     self.rankdata = self.loadPopularityData()

    def getEvaluation(self):
        return self.fulldata
    def getRank(self):
        return self.popularitydata


    def __init__(self):
        # build the full data

        self.fulldata = self.LoadRating()
        self.popularitydata = self.loadPopularityData()
        self.fullTrainData = self.fulldata.build_full_trainset()
        #build the full anti data test set
        self.fullAntiTestData = self.fullTrainData.build_anti_testset()
        self.fullTestData = self.fullTrainData.build_testset()

        #get 80% train data and 20% test data
        self.traindata, self.testdata = train_test_split(self.fulldata, test_size=0.2)


        #build leave-one-out cross validation
        self.LOO_Data = LeaveOneOut()
        for train, test in self.LOO_Data.split(self.fulldata):
            self.LOO_Train = train
            self.LOO_Test = test
        self.LOOAntiTest = self.LOO_Train.build_anti_testset()

        #pass the popularitydata
        self.rank = self.popularitydata

        #similarity used for diversity

        sim_options = {'name': 'cosine', 'user_based': False}  # compute  similarities between items
        self.sim_matrix = KNNBaseline(sim_options=sim_options)
        self.sim_matrix.fit(self.fullTrainData)

    #getter
    def GetFullTrainData(self):
        return self.fullTrainData

    def GetAntiTestData(self):
        return self.fullAntiTestData

    def GetAntiUserTestData(self,userId): #the same logic as the build_anti_test but for the spefic user
        trainset = self.fullTrainData
        temp = trainset.global_mean

        antiUserDataSet = []
        uidint = trainset.to_uid_inner(str(userId)) #find the specific user inner id
        user_watched_movies =set(x for (x,y) in trainset.ur[uidint]) #since int the train set, we use innter id
        antiUserDataSet+=[(trainset.to_raw_uid(uidint),trainset.to_raw_iid(i),temp) for i in trainset.all_items()
                          if i not in user_watched_movies] #since we find the data in the pandas later, we record the raw id
        return antiUserDataSet
    def GetFullTestData(self):
        return self.fullTestData

    def GetTrainData(self):
        return self.traindata

    def GetTestData(self):
        return self.testdata

    def GetLOOTrain(self):
        return self.LOO_Train
    def GetLOOTest(self):
        return self.LOO_Test
    def GetLOOAntiTestSet(self):
        return self.LOOAntiTest

    def GetPopularRankings(self):
        return self.rank

    def GetSimilarities(self):
        return self.sim_matrix
