import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV

# ======================== Step 1 : Load Data ========================

file_name = "train_set"

with open("input/" + file_name + "_data.npy", "rb") as np_file:
    data = np.load(np_file)
with open("input/names.npy", "rb") as np_file:
    col_names = np.load(np_file)

train_data = pd.DataFrame(data, columns=col_names)

print("=>=>=> Data loaded")

# ======================== Step 2 : Split Dataset ========================

train_one = data[data["label_cat"] == 1]
train_zero = data[data["label_cat"] == 0]

print("Labels rebalanced")

# ======================== Step 3 : Create Clusters ========================
estimator = MiniBatchKMeans(n_clusters=6, init="k-means++", max_iter=100, batch_size=100, verbose=0,
                            compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
                            n_init=3, reassignment_ratio=0.01)

params = {"n_clusters": range(1, 10)}
model = GridSearchCV(estimator=estimator, param_grid=params)

train_one_cluster = model_zero.fit_predict(train_one)

model_zero = MiniBatchKMeans(n_clusters=2, init="k-means++", max_iter=100, batch_size=100, verbose=0,
                             compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
                             n_init=3, reassignment_ratio=0.01)

train_zero_cluster = model_zero.fit_predict(train_zero)


