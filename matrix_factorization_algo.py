"""
This class contains all Matrix Factorization algorithms that we want to explore for our recommander system.
algorithms included:
1. SVD
2. SVDpp (SVD++): taking into account implicit ratings
3. PMF (Probabilistic Matrix Factorization): setting "biased" parameter to False
4. NMF (Non-negative Matrix Factorization)

for SVD and SVDpp, user and item factors are randomly initialized according to a normal distribution
--> can be tuned using the init_mean and init_std_dev parameters
"""

from surprise import SVD, SVDpp, NMF
from surprise import NormalPredictor
from surprise.model_selection import GridSearchCV


class MatrixFactorizationAlgo:
    # return a dictionary of algorithms; key: name of algo, val: algo object
    def generate_untuned_algorithms(self):
        algo = {'SVD': SVD(), 'PMF': SVD(biasd=False), 'SVDpp': SVDpp(), 'NMF': NMF()}
        print('Added SVD, PMF, SVD++, and NMF.')
        return algo


    def generate_tuned_algorithms(self, rating_data):
        algo = {}
        print('Tuning SVD parameters.')
        param_grid_svd = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
                      'n_factors': [50, 100]}
        # use GridSearchCVcomputes which (from surpise documentation)
        # computes accuracy metrics for an algorithm on various combinations of parameters, over a cross-validation procedure.
        # 1st param: algo we want to tune for
        # 2nd param: dictionary with algo parameters as keys and list of values as keys
        # default measures are measures=['rmse', 'mae'] default KFold is used with n_splits=5 (REPORT explain)
        grid_search_svd = GridSearchCV(SVD, param_grid_svd)

        grid_search_svd.fit(rating_data)

        # print the best RMSE
        print("best RMSE for SVD: ", grid_search_svd.best_score['rmse'])

        params = grid_search_svd.best_params['rmse']
        # print the best set of parameters
        print(params)

        SVD_tuned = SVD(n_epochs=params['n_epochs'], lr_all=params['lr_all'], n_factors=params['n_factors'])

        print('Tuning SVD++ parameters.')
        SVDpp_tuned = SVDpp() #todo
        algo = {'SVD_tuned' : SVD_tuned, 'SVDpp_tuned' : SVDpp_tuned}
        print('Added tuned SVD and SVD++.')
        return algo
