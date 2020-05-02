# each algorithm evaluation can take in an prediction algorithm,
# generate recommendations, and then report accuracy date
from surprise import accuracy
from collections import defaultdict

#from DataHandler import DataHandler

from DataHandler import DataHandler
import itertools as itr


class algorithm_eval:
    # constructor
    # algorithm: the prediction algorihtm we use
    # name: algorithm name
    NUM_DIGITS = 3

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    # returns a accuracy report including multiple metrics


    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm

    # get mean absolute error
    def MAE(self,predictions):
        return round(accuracy.mae(predictions, verbose=False), self.NUM_DIGITS)
        # return accuracy.mae(predictions, verbose=False)

    # get groot mean sqrt error:
    # penalize more when prediction way off, less when prediction close
    def RMSE(self,predictions):
        return round(accuracy.rmse(predictions, verbose=False), self.NUM_DIGITS)
        # return accuracy.rmse(predictions, verbose=False)

    # return a map w/
    #     key: user, value: a list of top n estRating movies (movieID, estRating)
    # TODO: n change to 10
    def getTopN(self, predictions, n=50, ratingCutOff=4.0):
        # create a map - key: userID, value: a list of (movieID, estRating)
        res = defaultdict(list)
        res2 = defaultdict(list) # list with actualRating


        for userID, movieID, actlRating, estRating, _ in predictions:
            # if estRating is larger than rating cut-off, add the movies w/
            # estRating to the topN list of the corresponding user
            if (estRating >= ratingCutOff):
                res[int(userID)].append((int(movieID), estRating))
                res2[int(userID)].append((int(movieID), estRating, actlRating))
        # for each user-list pair, sort the list by estRating, and keep top # N
        for userID, movieList in res.items():
            sorted(movieList, key=lambda x:x[1], reverse=True)
            res[userID] = movieList[0:n] # keep top N

        for userID, movieList in res2.items():
            sorted(movieList, key=lambda x:x[1], reverse=True)
            res2[userID] = movieList[0:n] # keep top N

        return res, res2

    # def hitRate(self, topNPred, leftOutData):
    #     return -1
    #
    # def cumulativeHitRate(self, topNPred, leftOutData, ratingCutOff=3.0):
    #     return -1
    #
    # def ratingHitRate(self, topNPred, leftOutPred):
    #     return -1
    #
    # def avrgReciprocalHitRank(self, topNPred, leftOutData):
    #     return -1

    # returns numHits / totalLeftOut
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # leftOutData: a list of left out data with high ratings from training set
    def hitRate(self,topNPred, leftOutData):
        # for each left out data, if the corresponding user has that movie in
        # its top N list, count it as a hit
        numHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            userID = int(data[0])
            movieID = int(data[1])

            # check whether left-out movie is in topN list of user
            for predMovieID, _ in topNPred[userID]:
                if (movieID == predMovieID):
                    numHits += 1
                    break
            # incremental total left out data
            totalLeftOut += 1
        return round(numHits / totalLeftOut, self.NUM_DIGITS)
        # return numHits / totalLeftOut

    # returns numHits / totalLeftOut, if the hits has ratings >= ratingCutOff
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # leftOutData: a list of left out data with high ratings from training set
    # ratingCutOff: if actual rating < ratingCutOff, does not count as hits
    # (deal with sparse data point? TODO: make sure I'm not lying)
    def cumulativeHitRate(self,topNPred, leftOutData, ratingCutOff=3.0):
        # for each left out data, if the corresponding user has that movie in
        # its top N list, count it as a hit
        numHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            actualRating = data[2]
            # if actual rating of left out movie >= cut off rating,
            # count hit if there exists one
            if (actualRating >= ratingCutOff):
                userID = int(data[0])
                movieID = int(data[1])

                # check whether left-out movie is in topN list of user
                for predMovieID, _ in topNPred[userID]:
                    if (movieID == predMovieID):
                        numHits += 1
                        break
                # incremental total left out data
                totalLeftOut += 1
        return round(numHits / totalLeftOut, self.NUM_DIGITS)
        #return numHits / totalLeftOut

    # returns numHits / totalLeftOut foe each rating seperately
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # leftOutData: a list of left out data with high ratings from training set
    def ratingHitRate(self,topNPred, leftOutPred):
        # key: rating, value: numHits / totalLeftOut corresponding to each rating
        numHits = defaultdict(float)
        totalLeftOut = defaultdict(float)

        # for each left out data, if the corresponding user has that movie in
        # its top N list, count it as a hit
        for data in leftOutPred:
            userID = int(data[0])
            movieID = int(data[1])
            actualRating = data[2]

            # check whether left-out movie is in topN list of user
            for predMovieID, _ in topNPred[userID]:
                if (movieID == predMovieID):
                    numHits[actualRating] += 1
                    break
            # incremental total left out data
            totalLeftOut[actualRating] += 1

        res = ""
        # arrange hit rates in increasing order of the corresponding ratings
        for rating in sorted(numHits.keys()):
            res += "{:<10} {:<10}\n".format(rating, round(numHits[rating]/totalLeftOut[rating], self.NUM_DIGITS))

        return res


    # returns rankedHits / totalLeftOut
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # @leftOutData: a list of left out data with high ratings from training set
    def avrgReciprocalHitRank(self, topNPred, leftOutData):
        sumRankedHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            userID = int(data[0])
            movieID = int(data[1])

            # check whether left-out movie is in topN list of user
            rank = 0
            for predMovieID, _ in topNPred[userID]:
                rank += 1 # for each movie in the top N list, increment its rank
                if (movieID == predMovieID):
                    sumRankedHits += 1.0 / rank
                    break
            # incremental total left out data
            totalLeftOut += 1
            return round(sumRankedHits / totalLeftOut, self.NUM_DIGITS)
            #return sumRankedHits / totalLeftOut

    # returns how diverse the recommendation is to users by using the simsAlgo
    # to compute the similarity between all pairs of recommendations for all users
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # @simsAlgo: the algorithm used to compute similarity between movies
    def diversity(self, topNPred, simsAlgo):
        numPairs = 0
        totalSim = 0
        simsMatrix = simsAlgo.compute_similarities()
        # for each user, get all possible pairs of the user's recommended movies
        for userID in topNPred.keys():
            movieList = topNPred[userID]
            for i in range(0, len(movieList)- 1):
                for j in range(i + 1, len(movieList)):
                    # for each pair, compute their similarity
                    # first convert movieID to innerID to more easily handle data
                    innerID_i = simsAlgo.trainset.to_inner_iid(str(movieList[i][0]))
                    innerID_j = simsAlgo.trainset.to_inner_iid(str(movieList[j][0]))

                    # get similarity between movie i and j, and add sim to total sim
                    totalSim += simsMatrix[innerID_i][innerID_j]
                    numPairs += 1

        # if no recommendation is generated
        if numPairs == 0:
            diversity = -1
        else:
            similarity = totalSim / numPairs
            diversity = 1 - similarity

        return round(diversity, self.NUM_DIGITS)
        #return diversity

    # returns the percentage of users whose recommendations actually have
    #         ratings greater than or equal to the predRatingTheshold
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # @numUsers: total number of user
    # @predRatingThreshold: the threshold where if a recommended movie has ratings
    #                      >= the threshold, the corresponding user counts as covered
    def userCoverage(self, topNPred, predRatingThreshold = 2.5):
        # for each user, check whether the user is covered
        numHits = 0
        numUsers = 0
        for userID in topNPred.keys():
            numUsers += 1
            # if there exists a movie in the user's recommendation whose rating
            # >= the predRatingTheshold, count as a hit
            for _, estRating in topNPred[userID]:
                if estRating >= predRatingThreshold:
                    numHits += 1
        return round(numHits / numUsers, self.NUM_DIGITS)
        #return numHits / numUsers

    # returns how new the recommended content is
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # @popularRankings: a dictionary with
    #                   key: movieID, value: popularity ranking of the movie (low value = high popularity)
    def novelty(self, topNPred, popularRankings):
        numMovies = 0
        totalNovelty = 0
        # for each user, get the novelty of the recommended movies
        for userID in topNPred.keys():
            for movieID, _ in topNPred[userID]:
                # add the novelty of the current movie
                totalNovelty += popularRankings[movieID]
                numMovies += 1

        # if no recommendation is generated
        if numMovies == 0:
            return -1
        return round(totalNovelty / numMovies, self.NUM_DIGITS)
        #return totalNovelty / numMovies

    # # returns number of recommended movies that are relevant / recommended movies
    # #    a movie is relevant if its actual rating is greater than a given threshold
    # # @topNPred: a dictionary w/ key: userID,
    # #                            value: list of top N ratings (moviesID, estRating, actualRating)
    # # @threshold: threshold for a movie to be considered as revelant
    # def precision(self, topNPred, threshold=2.5):
    #     numRelevant = 0
    #     numRecommend = 0
    #
    #     # for each user, get num of relevant movies
    #     for userID, movieList in topNPred.items():
    #         for _, _, actualRating in movieList: # if relevant
    #             if actualRating >= threshold:
    #                 numRelevant += 1
    #             numRecommend += 1
    #     return round(numRelevant / numRecommend, self.NUM_DIGITS)
    #
    # # returns num recommended movies that are relevant / relevant movies
    # #    a movie is relevant if its actual rating is greater than a given threshold
    # # @topNPred: a dictionary w/ key: userID,
    # #                            value: list of top N ratings (moviesID, estRating, actualRating)
    # # @completedPredictions: predictions for all movies and all users
    # # @threshold: threshold for a movie to be considered as revelant
    # def recall(self, topNPred, completedPredictions, threshold=3.5):
    #     numRelevant = 0
    #     numRecommendRelevant = 0
    #     # get all relevant movies
    #     for _, _, actualRating, _, _ in completedPredictions:
    #         if actualRating >= threshold:
    #             numRelevant += 1
    #
    #     for userID, movieList in topNPred.items():
    #         for _, _, actualRating in movieList:
    #             if actualRating >= threshold:
    #                 numRecommendRelevant += 1
    #
    #     return round(numRecommendRelevant / numRelevant , self.NUM_DIGITS)



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
            actualPredictions = self.algorithm.test(evaluationDataSet.GetLOOAntiTestSet())
            topNPredictions, topNPredictionsWithActual = self.getTopN(actualPredictions)
            metrics["HR"] = self.hitRate(topNPredictions, looPredictions)
            metrics["CHR"] =self.cumulativeHitRate(topNPredictions,looPredictions)
            metrics["RHR"] = self.ratingHitRate(topNPredictions,looPredictions)
            metrics["ARHR"] = self.avrgReciprocalHitRank(topNPredictions,looPredictions)

            metrics["Diversity"] = self.diversity(topNPredictions, evaluationDataSet.GetSimilarities())
            metrics["Coverage"] = self.userCoverage(topNPredictions)
            metrics["Novelty"] = self.novelty(topNPredictions, evaluationDataSet.GetPopularRankings())

            # metrics["Precision"] = self.precision(topNPredictionsWithActual)
            # metrics["Recall"] = self.recall(topNPredictionsWithActual, self.algorithm.test(evaluationDataSet.GetFullTestData()))
            # TODO: what the heck is this GetFullTestData?

        # Compute accuracy
        return metrics
