import csv

from EvaluationDataSet import EvaluationData
from algorithm_eval import algorithm_eval
class Evaluator:
    algos = []
    movie_id_to_name = {}
    movie = '../ml-latest-small/movies.csv'
    def __init__(self,dataset,rank):
        self.dataset = EvaluationData(dataset, rank)#a

    def Add_Algo(self,algorithm):
        alg = algorithm_eval(algorithm,algorithm.getName())
        self.algos.append(alg)

    def readMovieName(self):
        with open(self.movies, newline='', encoding='ISO-8859-1') as csvfile:
                Reader = csv.reader(csvfile)
                next(Reader)  #Skip header line
                for row in Reader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movie_id_to_name[movieID] = movieName

    def getMovieName(self,movieId):
        return self.movie_id_to_name[movieId]


    def print(self,TopN):
        result = {}
        for algo in self.algos:
            result[algo.getName()] = algo.evaluate(self.dataset,TopN)

        if(TopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "CHR","RHR", "ARHR", "FCP", "Diversity", "Coverage","Novelty"))
            for(name, metrics) in result.items():
                print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                metrics["Algorithm"],metrics["RMSE"], metrics["MAE"],
                metrics["HR"], metrics["CHR"],metrics["RHR"], metrics["ARHR"],metrics["FCP"], metrics["Diversity"],
                metrics["Coverage"],metrics["Novelty"]))
            else:
                print("{:<10} {:<10} {:<10}".format(metrics["Algorithm"],metrics["FCP"], metrics["RMSE"], metrics["MAE"] ))
        print("\n Note: \n")
        print("RMSE: Root Mean Squared Error")
        print("MAE: Mean Average Error")
        if(TopN):
            print("HR: Hit Rate")
            print("CHR: Cumulative Hit Rate")
            print("RHR: Rating Hit Rate ")
            print("ARHR: Average Rank Hit Rate")
            print("FCP: Fraction of Concordant Pairs")
            print("Diversity: 1-Similarity")
            print("Coverage: Ratio of users for whom recommendations above a certain threshold exist.")
            print("Novelty: Average popularity rank of recommended items.")

    def GenerateTopNRecs(self,movieList,testUser,N):
        for algo in self.algos:
            print("\n",algo.getName())
            #get full train data set
            print("\n Training Data:   ")
            fulltraindata = self.dataset.GetFullTrainData()
            algo.getAlgorithm().fit(fulltraindata)

            #get user anti test data set
            userAntiData = self.dataset.GetAntiUserTestData(testUser)
            predictions = algo.getAlgorithm().test(userAntiData)

            print("The Top", N, "Recommendations Are: ")
            recommendations = []
            for userID, movieID, realRating, estimatedRating,details in predictions:
                movieid_int = int(movieID)
                recommendations.append((movieid_int,estimatedRating))
            recommendations.sort(key = lambda  x:x[1],reverse=True)

            for rating in recommendations[:N]:
                print(movieList.getMovieName(rating[0]),rating[1])



















