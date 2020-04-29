import numpy as np

# UxF: user * feature matrix
# FxM: feature * movie matrix
# R: past ratings matrix
# U: number of users
# M: number of movies
# F: number of features
def svd(UxF, FxM, R, U, M, F):
    alpha = 0.002
    beta = 0.02
    repeat = 5000
    stopRate = 0.001

    # repeat steps
    for t in range(repeat):
        for u in range(U):
            for m in range(M):
                # if has rating, compute error
                if R[u][m] is not None:
                    err = R[u][m] - np.dot(UxF[u, :], FxM[:, m])

                    # tuning
                    for f in range(F):
                        UxF[u][f] = UxF[u][f] + alpha * (2 * err * FxM[f][m] - beta * UxF[u][f])
                        FxM[f][m] = FxM[f][m] + alpha * (2 * err * UxF[u][f] - beta * FxM[f][m])
        # after re-adjusting param Mtrx, compute new ratings
        newR = np.dot(UxF, FxM)
        errSum = 0
        for u in range(U):
            for m in range(M):
                if R[u][m] is not None:
                    # if has rating, compute error
                    errSum = errSum + pow(R[u][m] - np.dot(UxF[u, :], FxM[:, m]), 2)
                    
                    # TODO: regularization[???]
                    for f in range(F):
                        errSum = errSum + (beta / 2) * (pow(UxF[u][f], 2) + pow(FxM[f][m], 2))
        # if err is small enough for us to stop repeating
        if errSum < stopRate:
            break
    return UxF, FxM

    

def main():
    # num of features
    FList = np.array([5, 20, 50, 100])

    # ratings matrx
    R = [
            [None, 1, 1, 3, 1],
            [1, 2, None, 1, 3],
            [None, 1, 1, 3, None],
            [None, None, 5, 4, 4]
        ]

    R = np.array(R)

    U = len(R)      # num of users
    M = len(R[0])   # num of movies
    
    # randomly initialize 2 parameter matrices
    for f in range(len(FList)):
        F = FList[f]
        UxF = np.random.rand(U, F)
        FxM = np.random.rand(F, M)

        # get new 
        newUxF, newFxM = svd(UxF, FxM, R, U, M, F)
        newR = np.dot(newUxF, newFxM)

        print("-------------------number of features =", F, "---------------------")
        print(newR)
        print("-----------------END number of features =", F, "-------------------\n")

main()

