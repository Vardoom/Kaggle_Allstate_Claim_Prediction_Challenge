import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
import pickle


def main():

    # ======================== Step 1 : Load Data ========================

    file_name = "train_set"

    with open("input/" + file_name + "_data.npy", "rb") as np_file:
        data = np.load(np_file)
    with open("input/names.npy", "rb") as np_file:
        col_names = np.load(np_file)

    data = pd.DataFrame(data, columns=col_names)

    print("Data loaded")

    # ======================== Step 2 : Re-balance label ========================

    train_one = data[data["label_cat"] == 1]
    train_zero = data[data["label_cat"] == 0]
    train_data = train_one.append(train_zero.iloc[:train_one.shape[0] * 4, :])

    print("Labels rebalanced")

    # ======================== Step 3 : Construct Model ========================

    train_y_ord = train_data["label_ord"]
    train_data.drop("label_ord", axis=1, inplace=True)

    train_y = np.array(train_data["label_cat"], dtype=int)
    train_x = train_data.drop("label_cat", axis=1)

    estimator_0 = SVC(kernel="rbf", probability=True)

    params_0 = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]

    print("Model Build\nHyper-parameters tuning in progress")

    score_0 = make_scorer(roc_auc_score)
    score_1 = make_scorer(average_precision_score)

    model = GridSearchCV(estimator_0, param_grid=params_0, scoring={"sc_0": score_0, "sc_1": score_1}, refit="sc_1",
                         cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10), n_jobs=-1)
    model.fit(train_x, train_y)

    model_name = "SVM_1"

    with open("models/" + model_name + ".pkl", "wb") as model_file:
        pickle.dump(model, model_file)


if __name__ == "__main__":
    main()
