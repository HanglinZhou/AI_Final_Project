# each algorithm evaluation can take in an prediction algorithm,
# generate recommendations, and then report accuracy date
from surprise import accuracy
from collections import defaultdict
#from DataHandler import DataHandler

class algorithm_eval:
    # constructor
    # algorithm: the prediction algorihtm we use
    # name: algorithm name
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    # returns a accuracy report including multiple metrics


    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm

    # get fraction of concordant pairs
    # def FCP(self,predictions):
    #     return accuracy.fcp(predictions, verbose=False)
    def FCP(self,predictions):
        return -1

    # get mean absolute error
    def MAE(self,predictions):
        return accuracy.mae(predictions, verbose=False)

    # get groot mean sqrt error:
    # penalize more when prediction way off, less when prediction close
    def RMSE(self,predictions):
        return accuracy.rmse(predictions, verbose=False)

    # return a map w/
    # key: user, value: a list of top n estRating movies (movieID, estRating)
    def getTopN(self,predictions, n=10, ratingCutOff=4.0):
        # create a map - key: userID, value: a list of (movieID, estRating)
        res = defaultdict(list)

        for userID, movieID, actlRating, estRating, _ in predictions:
            # if estRating is larger than rating cut-off, add the movies w/
            # estRating to the topN list of the corresponding user
            if (estRating >= ratingCutOff):
                res[int(userID)].append((int(movieID), estRating))

        # for each user-list pair, sort the list by estRating, and keep top # N
        for userID, movieList in res.items():
            movieList.sort(key=lambda x:x[1], reverse=True)
            res[userID] = movieList[0:n] # keep top N

        return res

    def hitRate(self, topNPred, leftOutData):
        return -1

    def cumulativeHitRate(self, topNPred, leftOutData, ratingCutOff=3.0):
        return -1

    def ratingHitRate(self, topNPred, leftOutPred):
        return -1

    def avrgReciprocalHitRank(self, topNPred, leftOutData):
        return -1

    # returns numHits / totalLeftOut
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # leftOutData: a list of left out data with high ratings from training set
    # def hitRate(self,topNPred, leftOutData):
    #     # for each left out data, if the corresponding user has that movie in
    #     # its top N list, count it as a hit
    #     numHits, totalLeftOut = 0
    #     for data in leftOutData:
    #         userID = int(data[0])
    #         movieID = int(data[1])
    #
    #         # check whether left-out movie is in topN list of user
    #         for predMovieID, _ in topNPred[userID]:
    #             if (movieID == predMovieID):
    #                 numHits += 1
    #                 break
    #         # incremental total left out data
    #         totalLeftOut += 1
    #     return numHits / totalLeftOut
    #
    # # returns numHits / totalLeftOut, if the hits has ratings >= ratingCutOff
    # # @topNPred: a dictionary w/ key: userID,
    # #                            value: list of top N ratings (moviesID, estRating)
    # # leftOutData: a list of left out data with high ratings from training set
    # # ratingCutOff: if actual rating < ratingCutOff, does not count as hits
    # # (deal with sparse data point? TODO: make sure I'm not lying)
    # def cumulativeHitRate(self,topNPred, leftOutData, ratingCutOff=3.0):
    #     # for each left out data, if the corresponding user has that movie in
    #     # its top N list, count it as a hit
    #     numHits, totalLeftOut = 0
    #     for data in leftOutData:
    #         actualRating = data[2]
    #         # if actual rating of left out movie >= cut off rating,
    #         # count hit if there exists one
    #         if (actualRating >= ratingCutOff):
    #             userID = int(data[0])
    #             movieID = int(data[1])
    #
    #             # check whether left-out movie is in topN list of user
    #             for predMovieID, _ in topNPred[userID]:
    #                 if (movieID == predMovieID):
    #                     numHits += 1
    #                     break
    #             # incremental total left out data
    #             totalLeftOut += 1
    #     return numHits / totalLeftOut
    #
    # # returns numHits / totalLeftOut foe each rating seperately
    # # @topNPred: a dictionary w/ key: userID,
    # #                            value: list of top N ratings (moviesID, estRating)
    # # leftOutData: a list of left out data with high ratings from training set
    # def ratingHitRate(self,topNPred, leftOutPred):
    #     # key: rating, value: numHits / totalLeftOut corresponding to each rating
    #     numHits = defaultdict(float)
    #     totalLeftOut = defaultdict(float)
    #
    #     # for each left out data, if the corresponding user has that movie in
    #     # its top N list, count it as a hit
    #     for data in leftOutPred:
    #         userID = int(data[0])
    #         movieID = int(data[1])
    #         actualRating = data[2]
    #
    #         # check whether left-out movie is in topN list of user
    #         for predMovieID, _ in topNPred[userID]:
    #             if (movieID == predMovieID):
    #                 numHits[actualRating] += 1
    #                 break
    #         # incremental total left out data
    #         totalLeftOut[actualRating] += 1
    #
    #     res = ""
    #     # arrange hit rates in increasing order of the corresponding ratings
    #     for rating in numHits.keys().sort():
    #         res += "{} {}\n".format(rating, numHits[rating]/totalLeftOut[rating])
    #
    #     return res
    #
    #
    # # returns rankedHits / totalLeftOut
    # # @topNPred: a dictionary w/ key: userID,
    # #                            value: list of top N ratings (moviesID, estRating)
    # # leftOutData: a list of left out data with high ratings from training set
    # def avrgReciprocalHitRank(self,topNPred, leftOutData):
    #     sumRankedHits, totalLeftOut = 0
    #     for data in leftOutData:
    #         userID = int(data[0])
    #         movieID = int(data[1])
    #
    #         # check whether left-out movie is in topN list of user
    #         rank = 0
    #         for predMovieID, _ in topNPred[userID]:
    #             rank += 1 # for each movie in the top N list, increment its rank
    #             if (movieID == predMovieID):
    #                 sumRankedHits += 1.0 / rank
    #                 break
    #         # incremental total left out data
    #         totalLeftOut += 1
    #     return sumRankedHits / totalLeftOut

    def diversity(self,topNPred, leftOutPred):
        return -1

    def userCoverage(self,topNPred, numUsers, ratingThreshold):
        return -1

    def novelty(topNPred, rankings):
        return -1



    def evaluate(self, evaluationDataSet, TopN, n=10, verbose=True):
        metrics = {}
        if(verbose):
            print("Evaluating:")
        self.algorithm.fit(evaluationDataSet.GetTrainData())
        predictions_acc= self.algorithm.test(evaluationDataSet.GetTestData())

        metrics["RMSE"] = self.RMSE(predictions_acc) #how to call local function?
        #print("rmse", metrics["RMSE"])
        metrics["MAE"] = self.MAE(predictions_acc) #how to call local funciton?
        #print("mae", metrics['MAE'])
        if(TopN):
            # if(verbose):
            #     print("Evluating top n recommender: ")
                #train the model
            self.algorithm.fit(evaluationDataSet.GetLOOTrain())
            #Prepare for the left one out cross validation
            looPredictions = self.algorithm.test(evaluationDataSet.GetLOOTest())
            actuaPredictions = self.algorithm.test(evaluationDataSet.GetLOOAntiTestSet())
            #Prepare for top-N evaluations
            topNPredictions = self.getTopN(actuaPredictions)
            metrics["HR"] = self.hitRate(looPredictions,topNPredictions)
            metrics["CHR"] =self.cumulativeHitRate(topNPredictions,looPredictions)
            metrics["RHR"] = self.ratingHitRate(topNPredictions,looPredictions)
            metrics["ARHR"] = self.avrgReciprocalHitRank(topNPredictions,looPredictions)

            metrics["FCP"] = self.FCP(topNPredictions)
            metrics["Diversity"] = -1
            metrics["Coverage"] = -1
            metrics["Novelty"] =-1

        # Compute accuracy
        return metrics
