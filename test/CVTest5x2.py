from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import paired_ttest_5x2cv
import numpy as np
from scipy.stats import t as t_dist

# Importing the required libs
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import norm, chi2
from scipy.stats import t as t_dist
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold

# Libs implementations
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import proportion_difference
from mlxtend.evaluate import paired_ttest_kfold_cv
from mlxtend.evaluate import paired_ttest_resampled# Getting the wine data from sklearn
X, y = load_wine(return_X_y = True)# Instantiating the classification algorithms
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)# For holdout cases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



def wholeExample():
    X, y = iris_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    clf1 = LogisticRegression(random_state=1)
    clf2 = DecisionTreeClassifier(random_state=1, max_depth=1)

    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)
    print('Decision tree accuracy: %.2f%%' % (score2*100))

    t, p = paired_ttest_5x2cv(estimator1=clf1,
                              estimator2=clf2,
                              X=X, y=y,
                              random_seed=1)

    print('t statistic: %.3f' % t)
    print('p value: %.3f' % p)

    return 0


def five_two_statistic(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p_hat = (p1 + p2) / 2
    s = (p1 - p_hat) ** 2 + (p2 - p_hat) ** 2
    # t = p1[0] / np.sqrt(1 / 5. * sum(s))
    t = p2[0] / np.sqrt(1 / 5. * sum(s))

    p_value = t_dist.sf(np.abs(t), 5) * 2.  # 这里要取绝对值哦
    # p_value = t_dist.sf(t, 5) * 2.

    return t, p_value

def another_example():
    p_1 = []
    p_2 = []

    rng = np.random.RandomState(42)
    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=randint)

        rf.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        acc1 = accuracy_score(y_test, rf.predict(X_test))
        acc2 = accuracy_score(y_test, knn.predict(X_test))
        p_1.append(acc1 - acc2)

        rf.fit(X_test, y_test)
        knn.fit(X_test, y_test)
        acc1 = accuracy_score(y_train, rf.predict(X_train))
        acc2 = accuracy_score(y_train, knn.predict(X_train))
        p_2.append(acc1 - acc2)  # Running the test
    print("example: 5x2 CV Paired t-test")
    t, p = five_two_statistic(p_1, p_2)
    print(f"example: t statistic: {t}, p-value: {p}\n")


if __name__ == '__main__':
    another_example()

    distant_0_0 = np.array([90.3061, 91.8367, 90.3061, 93.3673, 91.8367])
    distant_0_1 = np.array([90.5371, 91.0486, 87.9795, 91.8159, 90.5371])

    distant_1_0 = np.array([94.5886, 94.5886, 95.997, 94.5886, 95.2557])
    distant_1_1 = np.array([94.362, 93.9169, 95.4006, 93.9169, 94.0653])

    p_1 = distant_1_0 - distant_0_0
    p_2 = distant_1_1 - distant_0_1

    print("5x2 CV Paired t-test")
    t, p = five_two_statistic(p_1, p_2)
    print(f"t statistic: {t}, p-value: {p}\n")




