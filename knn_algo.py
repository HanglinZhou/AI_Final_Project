from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import KNNWithMeans
from surprise import KNNBaseline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from EvaluationDataSet import EvaluationData

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
        clf = GridSearchCV(algo['bcKNN'], hyperparameters, cv=10)

        data = EvaluationData.GetTestData(EvaluationData)

        # Fit the model
        tunedbcKNN = clf.fit(data)
        # Print The value of best Hyperparameters
        print('bc KNN Best leaf_size:', tunedbcKNN.best_estimator_.get_params()['leaf_size'])
        print('bc KNN Best p:', tunedbcKNN.best_estimator_.get_params()['p'])
        print('bc KNN Best n_neighbors:', tunedbcKNN.best_estimator_.get_params()['n_neighbors'])

        algo['tunedbcKNN'] = tunedbcKNN

        # Use GridSearch
        clf = GridSearchCV(algo['wmKNN'], hyperparameters, cv=10)

        # Fit the model
        tunedwmKNN = clf.fit(data)
        # Print The value of best Hyperparameters
        print('wm KNN Best leaf_size:', tunedwmKNN.best_estimator_.get_params()['leaf_size'])
        print('wm KNN Best p:', tunedwmKNN.best_estimator_.get_params()['p'])
        print('wm KNN Best n_neighbors:', tunedwmKNN.best_estimator_.get_params()['n_neighbors'])

        algo['tunedwmKNN'] = tunedwmKNN

        # Use GridSearch
        clf = GridSearchCV(algo['wzKNN'], hyperparameters, cv=10)

        # Fit the model
        tunedwzKNN = clf.fit(data)
        # Print The value of best Hyperparameters
        print('wz KNN Best leaf_size:', tunedwzKNN.best_estimator_.get_params()['leaf_size'])
        print('wz KNN Best p:', tunedwzKNN.best_estimator_.get_params()['p'])
        print('wz KNN Best n_neighbors:', tunedwzKNN.best_estimator_.get_params()['n_neighbors'])

        algo['tunedwzKNN'] = tunedwzKNN

        # Use GridSearch
        clf = GridSearchCV(algo["blKNN"], hyperparameters, cv=10)

        # Fit the model
        tunedblKNN = clf.fit(data)
        # Print The value of best Hyperparameters
        print('bl KNN Best leaf_size:', tunedblKNN.best_estimator_.get_params()['leaf_size'])
        print('bl KNN Best p:', tunedblKNN.best_estimator_.get_params()['p'])
        print('bl KNN Best n_neighbors:', tunedblKNN.best_estimator_.get_params()['n_neighbors'])

        algo['tunedblKNN'] = tunedblKNN

        return algo

