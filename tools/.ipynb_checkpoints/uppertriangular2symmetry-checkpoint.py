import numpy as np


# 这个方法是将一个上三角矩阵转化为一个对称矩阵
# 下对焦矩阵的话，这个方法怕是行不通哦。因此你可以先进行转置


def up2sym(matrixX, zero_diag=False):
    matrixX = np.triu(matrixX)  # 保留其上三角部分
    # 注意，要减去一次对角线上的元素。因为上三角cov，和下三角cov.T在进行相加时会把主对角线上的元素相加两次。
    # 或减去两倍对角线元素，可得到对角线为0的对称矩阵
    matrixX += matrixX.T - np.diag(matrixX.diagonal())  # 将上三角”拷贝”到下三角部分
    if zero_diag:
        matrixX -= np.diag(np.diagonal(matrixX))
        # matrixX += matrixX.T - 2 * np.diag(matrixX.diagonal())  # 对角线元素归 0 操作
    return matrixX



def low2sym(matrixX, zero_diag=False):
    matrixX = np.tril(matrixX)  # 保留其下三角部分
    # 注意，要减去一次对角线上的元素。因为下三角cov，和上三角cov.T在进行相加时会把主对角线上的元素相加两次。
    # 或减去两倍对角线元素，可得到对角线为0的对称矩阵
    matrixX += matrixX.T - np.diag(matrixX.diagonal())  # 将下三角”拷贝”到上三角部分
    if zero_diag:
        matrixX -= np.diag(np.diagonal(matrixX))
        # matrixX += matrixX.T - 2 * np.diag(matrixX.diagonal())  # 对角线元素归 0 操作
    return matrixX


if __name__ == '__main__':
    oX = np.arange(25).reshape(5, 5)
    print(oX)

    X = low2sym(oX, True)
    print(X)

    test = np.shape(X)
    test = oX == X  # 测试
    print(test)

    test = X.T == X  # 测试
    print(test)
