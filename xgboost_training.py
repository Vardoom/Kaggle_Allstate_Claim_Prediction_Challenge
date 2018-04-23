import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt


def predict_results(alg, d_train_x, d_train_y):
    # Predict training set:
    d_train_predictions = alg.predict(d_train_x)
    d_train_predictions_prob = alg.predict_proba(d_train_x)[:, 1]

    # Print model report:
    print("Model Report")
    print("Accuracy : {} %".format(accuracy_score(d_train_y, d_train_predictions) * 100))
    print("AUC Score (Train): {} %".format(roc_auc_score(d_train_y, d_train_predictions_prob) * 100))
    print("AP Score (Train): {} %".format(average_precision_score(d_train_y, d_train_predictions_prob) * 100))

    return d_train_predictions, d_train_predictions_prob


def model_fit(alg, d_train_x, d_train_y, cv_folds=5, early_stopping_rounds=50):
    xgb_param = alg.get_xgb_params()
    xg_train = xgb.DMatrix(d_train_x, label=d_train_y)
    cv_result = xgb.cv(xgb_param, xg_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                       metrics='map', early_stopping_rounds=early_stopping_rounds, folds=KFold)

    alg.set_params(n_estimators=cv_result.shape[0])

    # Fit the algorithm on the data
    alg.fit(d_train_x, d_train_y, eval_metric='map')

    return cv_result


def load_model(mod_name):
    with open("models/" + mod_name + ".pkl", "rb") as mod_file:
        cur_model = pickle.load(mod_file)
        cur_cv_res = pickle.load(mod_file)
    return cur_model, cur_cv_res


def save_model(mod_name, alg, cv_res):
    with open("models/" + mod_name + ".pkl", "wb") as mod_file:
        pickle.dump(alg, mod_file)
        pickle.dump(cv_res, mod_file)


def model_grid_fit(est, param, d_train_x, d_train_y):
    cur_model = GridSearchCV(estimator=est, param_grid=param, scoring=["roc_auc", "average_precision"], n_jobs=4,
                             iid=False, cv=5, return_train_score=True, refit="average_precision")
    cur_model.fit(d_train_x, d_train_y)
    return cur_model, cur_model.cv_results_


def display_param(mod_name, cv_res, alg, grid_cv=False):
    if grid_cv:
        print(model_name)
        print("CV Results:")
        for key in cv_results.keys():
            print("{} : {}".format(key, cv_results[key]))
        print("Best parameters: {}".format(model.best_params_))
    else:
        print(mod_name)
        print("CV Results:\n{}".format(cv_res))
        print("Parameters:")
        p = alg.get_xgb_params()
        for key in p.keys():
            print("{}: {}".format(key, p[key]))


def plot_curves(mod_list, d_train_y):
    plt.figure(figsize=(10, 15))

    plt.subplot(211)
    for mod_k in mod_list.keys():
        roc_in, roc_out, _ = roc_curve(d_train_y, mod_list[mod_k][2])
        plt.plot(roc_in, roc_out, label="ROC curve of {}".format(mod_k))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title("ROC Curve")

    plt.subplot(212)
    for mod_k in mod_list.keys():
        precision, recall, _ = precision_recall_curve(d_train_y, mod_list[mod_k][2])
        plt.plot(precision, recall, label="RP curve of {}".format(mod_k))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend(loc="upper right")
    plt.title("RP Curve")

    plt.show()

# ======================== Step 1 : Load Data ========================

file_name = "train_set"

with open("input/" + file_name + "_data.npy", "rb") as np_file:
    data = np.load(np_file)
with open("input/names.npy", "rb") as np_file:
    col_names = np.load(np_file)

train_data = pd.DataFrame(data, columns=col_names)
train_y_ord = train_data["label_ord"]
train_data.drop("label_ord", axis=1, inplace=True)
train_y = train_data["label_cat"]
train_x = train_data.drop("label_cat", axis=1)

print("=>=>=> Data loaded")

# ======================== Step 2 : Construct Model ========================

model_list = dict()

# ======================== Step 2.1 : Test 1 ========================
print("=>=>=> Launching test 1")
model_name = "xgboost_1"
test = True

if test:
    # Define the model
    model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                          colsample_bytree=0.8, objective="binary:logistic", n_jobs=8, scale_pos_weight=1,
                          random_state=42, silent=False, kwargs={"tree_method": "gpu_hist"})

    # Get the results
    cv_results = model_fit(alg=model, d_train_x=train_x, d_train_y=train_y)

    # Save the model
    save_model(mod_name=model_name, alg=model, cv_res=cv_results)

else:
    # Load the model
    model, cv_results = load_model(mod_name=model_name)

# Print parameters
display_param(mod_name=model_name, cv_res=cv_results, alg=model, grid_cv=False)

# Make predictions
train_pred, train_prob = predict_results(alg=model, d_train_x=train_x, d_train_y=train_y)

# Retrieve parameters
n_estimators = model.get_xgb_params()["n_estimators"]

model_list[model_name] = [model, train_pred, train_prob]

print("Test 1 Over")


# ======================== Step 2.2 : Test 2 ========================
print("=>=>=> Launching test 2")
model_name = "xgboost_2"
test = True

if test:
    # Define the model
    estimator = XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=1, min_child_weight=0, gamma=0,
                              subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', n_jobs=4,
                              scale_pos_weight=1, random_state=27, silent=False, kwargs={"tree_method": "gpu_hist"})
    parameters = {"max_depth": range(1, 10, 2), "min_child_weight": range(0, 10, 2)}

    # Get the results
    model, cv_results = model_grid_fit(est=estimator, param=parameters, d_train_x=train_x, d_train_y=train_y)

    # Save the model
    save_model(mod_name=model_name, alg=model, cv_res=cv_results)

else:
    # Load the model
    model, cv_results = load_model(mod_name=model_name)

# Print parameters
display_param(mod_name=model_name, cv_res=cv_results, alg=model, grid_cv=True)

# Make predictions
train_pred, train_prob = predict_results(alg=model, d_train_x=train_x, d_train_y=train_y)

# Retrieve parameters
max_depth = model.best_params_["max_depth"]
min_child_weight = model.best_params_["min_child_weight"]

model_list[model_name] = [model, train_pred, train_prob]

print("Test 2 Over")


# Plot final results
# plot_curves(model_list, train_y)
