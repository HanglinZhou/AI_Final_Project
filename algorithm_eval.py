# each algorithm evaluation can take in an prediction algorithm,
# generate recommendations, and then report accuracy date
from surprise import accuracy
from collections import defaultdict

class algorithm_eval:
    # constructor
    # algorithm: the prediction algorihtm we use
    # name: algorithm name
    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    # returns a accuracy report including multiple metrics
    def evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # Compute accuracy
        return metrics

    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm

    # get fraction of concordant pairs
    def FCP(predictions):
        return accuracy.fcp(predictions, verbose=False)

    # get mean absolute error
    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    # get groot mean sqrt error:
    # penalize more when prediction way off, less when prediction close
    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    # return a map w/
    # key: user, value: a list of top n estRating movies (movieID, estRating)
    def getTopN(predictions, n=10, ratingCutOff=4.0):
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
            res[userID] = res[0:N] # keep top N

        return res

    # returns numHits / totalLeftOut
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # leftOutData: a list of left out data with high ratings from training set
    def hitRate(topNPred, leftOutData):
        # for each left out data, if the corresponding user has that movie in
        # its top N list, count it as a hit
        numHits, totalLeftOut = 0
        for data in leftOutData:
            userID = int(data[0])
            movieID = int(data[1])

            # check whether movie is in topN list of user
            for predMovieID, _ in topNPred[userID]:
                if (movieID == predMovieID):
                    numHits = numHits + 1
            # incremental total left out data
            totalLeftOut = totalLeftOut + 1
        return numHits / totalLeftOut

    def cumulativeHitRate(topNPred, leftOutPred, ratingCutOff=0):
        return -1

    def ratingHitRate(topNPred, leftOutPred):
        return -1

    def avrgReciprocalHitRank(topNPred, leftOutPred):
        return -1

    def diversity(topNPred, leftOutPred):
        return -1

    def userCoverage(topNPred, numUsers, ratingThreshold):
        return -1

    def novelty(topNPred, rankings):
        return -1
