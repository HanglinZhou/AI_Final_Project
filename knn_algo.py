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
from surprise.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from surprise.model_selection import cross_validate

import numpy as np
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from surprise import Reader

class knn:
    # returns the dictionary containing all knn algorithms
    def generate_knn(self,rating_data):

        algo = {}
        bcKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
        algo['bcKNN'] = bcKNN

        wmKNN = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})
        algo['wmKNN'] = wmKNN

        wzKNN = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': True})
        algo['wzKNN'] = wzKNN

        blKNN = KNNBaseline(sim_options={'name': 'cosine', 'user_based': True})
        algo['blKNN'] = blKNN


        # tune param for knnBaseline, since it has best accuracy
        param_grid_bl = {'k': [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]}
        best_params_bl = self.tune_and_find_parameter('blKNN', KNNBaseline, rating_data, param_grid_bl)

        blKNN_tuned = KNNBaseline(k=best_params_bl['k'])
        algo.update({'blKNN_tuned': blKNN_tuned})

        return algo


    # returns the best parameters after tuning
    def tune_and_find_parameter(self,algo_name, algo, rating_data,param_grid):

        print("tuning for", algo_name, "hyperparameters")

        # algo: algo class name
        grid_search = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'])
        grid_search.fit(rating_data)

        print('best RMSE for ', algo_name, ' ', grid_search.best_score['rmse'])

        best_params = grid_search.best_params['rmse']
        # print the best set of parameters
        print("best params:", best_params)
        return best_params
