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


class MatrixFactorizationAlgo:


