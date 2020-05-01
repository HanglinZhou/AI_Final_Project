#import self as self
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline
from surprise import Trainset
class EvaluationData:

    def __int__(self, fulldata, popularitydata):

        # build the full data
        self.fullTrainData = fulldata.build_full_trainset()
        #build the full anti data test set
        self.fullAntiTestData = fulldata.build_anti_testset()

        #get 80% train data and 20% test data
        self.traindata, self.testdata = train_test_split(fulldata, test_size=0.2)


        #build leave-one-out cross validation
        self.LOO_Data = LeaveOneOut()
        for train, test in self.LOO_Data.split(fulldata):
            self.LOO_Train = train
            self.LOO_Test = test
        self.LOOAntiTest = self.LOO_Train.build_anti_testset()

        #pass the popularitydata
        self.rank = popularitydata

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

    def rank(self):
        return self.rank

    def GetSimilarities(self):
        return self.sim_matrix