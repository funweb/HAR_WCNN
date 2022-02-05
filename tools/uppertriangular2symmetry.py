import numpy as np


def up2sym(matrixX, zero_diag=False):
    matrixX = np.triu(matrixX)
    matrixX += matrixX.T - np.diag(matrixX.diagonal()) 
    if zero_diag:
        matrixX -= np.diag(np.diagonal(matrixX))
    return matrixX



def low2sym(matrixX, zero_diag=False):
    matrixX = np.tril(matrixX) 
    matrixX += matrixX.T - np.diag(matrixX.diagonal()) 
    if zero_diag:
        matrixX -= np.diag(np.diagonal(matrixX))
    return matrixX


if __name__ == '__main__':
    oX = np.arange(25).reshape(5, 5)
    print(oX)

    X = low2sym(oX, True)
    print(X)

    test = np.shape(X)
    test = oX == X  # test
    print(test)

    test = X.T == X  # test
    print(test)
