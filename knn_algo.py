"""
This class contains all KNN algorithms that we want to explore for our recommander system.
algorithms included:

1. k-NNBasic
2. k-NNWithMeans: taking into account the mean ratings of each user
3. k-NNWithZ-Score: taking into account the z-score normalization of each user
4. k-NNBaseline: taking into account a baseline rating

"""

from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import KNNWithMeans
from surprise import KNNBaseline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from surprise.model_selection import cross_validate
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from surprise import Reader

class knn:

    # Return knn algorithms in sequence of
    def untuned_knn_algo(self):

        algo = {}
        # User-based KNN cosine similarity
        bcKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
        algo['bcKNN'] = bcKNN

        wmKNN = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})
        algo['wmKNN'] = wmKNN

        wzKNN = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': True})
        algo['wzKNN'] = wzKNN

        blKNN = KNNBaseline(sim_options={'name': 'cosine', 'user_based': True})
        algo['blKNN'] = blKNN

        return algo

    # tuning by k-fold using cross-validation
    def knnBasic_tune(self,ratings,str):

        # creating odd list of K for KNN
        neighbors = list(range(1, 55, 4))

        # empty list that will hold cv scores
        cv_scores = []

        # empty list that will hold RMSE mean train scores
        rmse_train_scores = []

        # empty list that will hold MAE mean train scores
        mae_train_scores = []

        print("tuning " + str)

        # perform 10-fold cross validation
        for k in neighbors:
            algo = self.get_knn_algo(str,k)
            test = cross_validate(algo, ratings, measures=['RMSE', 'MAE'], cv=5, verbose = False)
            cv_scores.append(test)

            # get mean train rmse
            rmse = test['test_rmse']
            rmse_train_scores.append(np.sum(rmse)/5)

            # get mean train mae
            mae = test['test_mae']
            mae_train_scores.append(np.sum(mae)/5)

        return (cv_scores, rmse_train_scores, mae_train_scores)


    def analyze_knn_model(self,ratings, name):
        (cv, rmse, mae) = self.knnBasic_tune(ratings,name)
        neighbors = list(range(1, 15))

        # plot rmse score
        plt.plot(neighbors, rmse)
        plt.xlabel('Value of K for ' + name)
        plt.ylabel('rmse')
        plt.show()

        # plot mae score
        plt.plot(neighbors, mae)
        plt.xlabel('Value of K for ' + name)
        plt.ylabel('mae')
        plt.show()

        # contruct knn algo with smallest rmse score
        k = rmse.index(min(rmse)) + 1
        tuned_algo = self.get_knn_algo(name, k)

        return tuned_algo


    def get_knn_algo(self,str,k):
        if(str == 'bcKNN'):
            return KNNBasic(k = k, min = k, sim_options={'name': 'cosine', 'user_based': True},verbose = False)
        elif(str == 'wmKNN'):
            return KNNWithMeans(k = k, min = k, sim_options={'name': 'cosine', 'user_based': True}, verbose = False)
        elif(str == 'wzKNN'):
            return KNNWithZScore(k = k, min = k, sim_options={'name': 'cosine', 'user_based': True}, verbose = False)
        else:
            return KNNBaseline(k = k, min = k, sim_options={'name': 'cosine', 'user_based': True}, verbose = False)


    # # lines call at recommendation engine
    # # call KNNalgo generate algo
    # knn_algo = knn.untuned_knn_algo()
    #
    # # tune knn algo
    # tuned_knn_algo = {}
    # best_k = {}
    # for key in knn_algo:
    #     best_k[key], tuned_knn_algo[key] = knn.analyze_knn_model(ratings, key)