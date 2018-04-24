import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def grid_cv(cross_split, clust_nb, d_train_y):
    inertia_list = np.zeros(clust_nb)
    estimators = list()
    params = dict()

    params["n_clusters"] = 1
    est = MiniBatchKMeans(**params)
    est.fit(d_train_y)
    inertia_list[0] = est.inertia_
    estimators.append(est)
    print("Iteration: {}: n_clusters: {}, inertia: {}".format(0, 0, inertia_list[0]))

    params["n_clusters"] = 2
    est = MiniBatchKMeans(**params)
    est.fit(d_train_y)
    inertia_list[1] = est.inertia_
    estimators.append(est)
    print("Iteration: {}: n_clusters: {}, inertia: {}".format(1, 1, inertia_list[1]))

    i = 2
    while i < clust_nb and inertia_list[i - 2] > inertia_list[i - 1]:
        params["n_clusters"] = i
        est = MiniBatchKMeans(**params)
        est.fit(d_train_y)
        inertia_list[i] = est.inertia_
        estimators.append(est)
        print("Iteration: {}: n_clusters: {}, inertia: {}".format(i, i, inertia_list[i]))
        i += 1

    return estimators[i - 1]


def over_sampling(ov_s, num):
    ov_s_l = ov_s.shape[0]
    ov_res = ov_s.copy()
    sample_to_create = num - ov_s_l
    print("Need to resample: {} elements".format(sample_to_create))
    for id in range(sample_to_create):
        if id % 1000 == 0:
            print("Step {}/{}".format(id, sample_to_create))
        s = np.random.randint(low=0, high=ov_s_l)
        ov_res.append(ov_s.iloc[s, :])
    return ov_res


def main():

    # ======================== Step 1 : Load Data ========================

    file_name = "train_set"

    with open("input/" + file_name + "_data.npy", "rb") as np_file:
        data = np.load(np_file)
    with open("input/names.npy", "rb") as np_file:
        col_names = np.load(np_file)

    train_data = pd.DataFrame(data, columns=col_names)

    print("=>=>=> Data loaded")

    # ======================== Step 2 : Split Dataset ========================

    train_one = train_data[train_data["label_cat"] == 1].copy()
    train_zero = train_data[train_data["label_cat"] == 0].copy()

    print("=>=>=> Dataset splitted")

    # ======================== Step 3 : Create Clusters ========================

    estimator_zero = MiniBatchKMeans(n_clusters=8)
    train_zero_clusters = estimator_zero.fit_predict(train_zero)
    train_zero["cluster"] = train_zero_clusters
    l = np.unique(train_zero_clusters, return_counts=True)[1].max()

    estimator_one = MiniBatchKMeans(n_clusters=2)
    train_one_clusters = estimator_one.fit_predict(train_one)
    train_one["cluster"] = train_one_clusters

    print("=>=>=> Clusters Created")

    # ======================== Step 4 : Re-sample ========================

    col_names = train_one.columns
    final_train_data = pd.DataFrame(data=None, columns=col_names)

    for i in range(8):
        ov_sample = train_zero[train_zero["cluster"] == i].copy()
        ov_sample = over_sampling(ov_sample, l)
        final_train_data = final_train_data.append(ov_sample)

    for i in range(2):
        ov_sample = train_one[train_one["cluster"] == i].copy()
        ov_sample = over_sampling(ov_sample, int(2 * l / 3))
        final_train_data = final_train_data.append(ov_sample)

    # ======================== Step 5: Save Data ========================

    with open("input/" + file_name + "_sampled_data.npy", mode='wb') as numpy_file:
        np.save(numpy_file, final_train_data)
    with open("input/names.npy", mode='wb') as numpy_file:
        np.save(numpy_file, final_train_data.columns)
    print("File saved in numpy format")


if __name__ == "__main__":
    main()
