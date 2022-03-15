import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

target = np.random.randint(1, 5, 100)

# Choose seeds for each 2-fold iterations
seeds = [13, 51, 137, 24659, 347]
# Initialize the score difference for the 1st fold of the 1st iteration
p_1_1 = 0.0
# Initialize a place holder for the variance estimate
s_sqr = 0.0
# Initialize scores list for both classifiers
scores_1 = []
scores_2 = []
diff_scores = []
# Iterate through 5 2-fold CV
for i_s, seed in enumerate(seeds):
    # Split the dataset in 2 parts with the current seed
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # Initialize score differences
    p_i = np.zeros(2)
    # Go through the current 2 fold
    for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        # Split the data
        print(len(trn_idx), len(val_idx))
        pass