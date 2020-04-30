from surprise import accuracy

class recommender_metrics:
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

    def getTopN(predictions):
        return []

    def hitRate(topNPred, leftOutPred):
        return -1

    def cumulativeHitRate(topNPred, leftOutPred, ratingCutOff=0):
        return -1

    def ratingHitRate(topNPred, leftOutPred):
        return -1

    def avrgReciprocalHitRank(topNPred, leftOutPred):
        return -1

    def diversity(topNPred, leftOutPred):
        return 0

    def userCoverage(topNPred, numUsers, ratingThreshold):
        return 0

    def novelty(topNPred, rankings):
        return 0
