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
from EvaluationDataSet import EvaluationData
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

class knn:

    # Return knn algorithms in sequence of
    def UntunedknnAlgorithms():

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

    # tuning by k-fold cross-validation
    def KnnAlgo():

        algo = knn.UntunedknnAlgorithms()

        # List Hyperparameters that we want to tune.
        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]
        # Convert to dictionary
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        # Use GridSearch
        bcKNN = GridSearchCV(algo['bcKNN'],param_grid=hyperparameters,scoring='accuracy',cv=10)

        # get data
        x_train = knn.data_split()

        # Fit the model
        bcKNN.fit(x_train)
        # # Print The value of best Hyperparameters
        # print('bc KNN Best leaf_size:', tunedbcKNN.best_estimator_.get_params()['leaf_size'])
        # print('bc KNN Best p:', tunedbcKNN.best_estimator_.get_params()['p'])
        # print('bc KNN Best n_neighbors:', tunedbcKNN.best_estimator_.get_params()['n_neighbors'])

        algo['tunedbcKNN'] = bcKNN

        # # Use GridSearch
        # clf = GridSearchCV(algo['wmKNN'], hyperparameters, cv=10)
        #
        # # Fit the model
        # tunedwmKNN = clf.fit(x_train)
        # # Print The value of best Hyperparameters
        # print('wm KNN Best leaf_size:', tunedwmKNN.best_estimator_.get_params()['leaf_size'])
        # print('wm KNN Best p:', tunedwmKNN.best_estimator_.get_params()['p'])
        # print('wm KNN Best n_neighbors:', tunedwmKNN.best_estimator_.get_params()['n_neighbors'])
        #
        # algo['tunedwmKNN'] = tunedwmKNN
        #
        # # Use GridSearch
        # clf = GridSearchCV(algo['wzKNN'], hyperparameters, cv=10)
        #
        # # Fit the model
        # tunedwzKNN = clf.fit(x_train)
        # # Print The value of best Hyperparameters
        # print('wz KNN Best leaf_size:', tunedwzKNN.best_estimator_.get_params()['leaf_size'])
        # print('wz KNN Best p:', tunedwzKNN.best_estimator_.get_params()['p'])
        # print('wz KNN Best n_neighbors:', tunedwzKNN.best_estimator_.get_params()['n_neighbors'])
        #
        # algo['tunedwzKNN'] = tunedwzKNN
        #
        # # Use GridSearch
        # clf = GridSearchCV(algo["blKNN"], hyperparameters, cv=10)
        #
        # # Fit the model
        # tunedblKNN = clf.fit(x_train)
        # # Print The value of best Hyperparameters
        # print('bl KNN Best leaf_size:', tunedblKNN.best_estimator_.get_params()['leaf_size'])
        # print('bl KNN Best p:', tunedblKNN.best_estimator_.get_params()['p'])
        # print('bl KNN Best n_neighbors:', tunedblKNN.best_estimator_.get_params()['n_neighbors'])
        #
        # algo['tunedblKNN'] = tunedblKNN

        return algo
    
    def data_split():
        data_path = Path("./data/movieLens/")
        ratings = pd.read_csv(data_path / "ratings.csv")
        # Split data into training and testing.
        x_train, x_test = train_test_split(ratings, test_size=0.2)
        return(x_train)
        