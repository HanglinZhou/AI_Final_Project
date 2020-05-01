"""
This class contains all Matrix Factorization algorithms that we want to explore for our recommander system.
algorithms included:

1. SVD
2. SVDpp (SVD++): taking into account implicit ratings
3. PMF (Probabilistic Matrix Factorization): setting "biased" parameter to False
4. NMF (Non-negative Matrix Factorization)

#todo: more tuning algo & param
We also want to tune our chosen algorithms (SVD & SVDpp) and here is a list of parameters we are interested:
1. n_factors: The number of factors. Default is 100.
2. n_epochs: The number of iteration of the SGD procedure. Default is 20
3. lr_all – The learning rate for all parameters. Default is 0.005.
4. reg_all – The regularization term for all parameters. Default is 0.02.

If needed, instead of tuning lr_all and reg_all we can do:
lr_bu – The learning rate for 𝑏𝑢. Takes precedence over lr_all if set. Default is None.
lr_bi – The learning rate for 𝑏𝑖. Takes precedence over lr_all if set. Default is None.
lr_pu – The learning rate for 𝑝𝑢. Takes precedence over lr_all if set. Default is None.
lr_qi – The learning rate for 𝑞𝑖. Takes precedence over lr_all if set. Default is None.
reg_bu – The regularization term for 𝑏𝑢. Takes precedence over reg_all if set. Default is None.
reg_bi – The regularization term for 𝑏𝑖. Takes precedence over reg_all if set. Default is None.
reg_pu – The regularization term for 𝑝𝑢. Takes precedence over reg_all if set. Default is None.
reg_qi – The regularization term for 𝑞𝑖. Takes precedence over reg_all if set. Default is None.

"""

from surprise import SVD, SVDpp, NMF
from surprise.model_selection import GridSearchCV


class MatrixFactorizationAlgo:
    # return a dictionary of algorithms; key: name of algo, val: algo object
    def generate_algorithms(self, rating_data):
        # here we separate untuned and tuned algo as it might take a really long time on tuning,
        # it's easier to comment out the tuning part if needed
        algo = {'SVD': SVD(), 'PMF': SVD(biased=False), 'SVD++': SVDpp(), 'NMF': NMF()}
        print('Generated algo object for SVD, PMF, SVD++, and NMF, tuned SVD, and tuned SVD++.')

        print('Tuning SVD parameters: ')
        best_params_svd = self.tune_and_find_param('SVD_tuned', SVD, rating_data)
        SVD_tuned = SVD(n_factors = best_params_svd['n_factors'], n_epochs = best_params_svd['n_epochs'],
                        lr_all = best_params_svd['lr_all'], reg_all = best_params_svd['reg_all'])

        print('Tuning SVD++ parameters: ')
        best_params_svdpp = self.tune_and_find_param('SVD++_tuned', SVDpp, rating_data)

        SVDpp_tuned = SVDpp(n_factors = best_params_svdpp['n_factors'], n_epochs = best_params_svdpp['n_epochs'],
                        lr_all = best_params_svdpp['lr_all'], reg_all = best_params_svdpp['reg_all'])

        algo['SVD_tuned'] =  SVD_tuned
        algo['SVD++_tuned'] = SVDpp_tuned
        print('Generated algo object for tuned SVD and tuned SVD++.')

        return algo


    def tune_and_find_param(self, algo_name, algo, rating_data):
        param_grid = {'n_factors': [10, 200], 'n_epochs': [20, 50], 'lr_all': [0.001, 0.020],
                          'reg_all': [0.010, 0.030]}
        # use GridSearchCVcomputes which (from surpise documentation)
        # computes accuracy metrics for an algorithm on various combinations of parameters, over a cross-validation procedure.
        grid_search = GridSearchCV(algo, param_grid)

        grid_search.fit(rating_data)

        # print the best RMSE
        print('best RMSE for ', algo_name, ' ', grid_search.best_score['rmse'])

        best_params = grid_search.best_params['rmse']
        # print the best set of parameters
        print(best_params)
        return best_params

