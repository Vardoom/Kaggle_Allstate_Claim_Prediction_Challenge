import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer


# ======================== Useful functions ========================

def int2bin(i, length):
    return ("{0:{fill}" + str(length) + "b}").format(i, fill='0')


def binary_encoding(column, data):
    unique_list = data[column].unique()
    col_num = len(bin(unique_list.shape[0])) - 2
    new_columns = list()
    key_dict = dict()
    for unique_key in unique_list:
        key_dict[unique_key] = True
    i = 0
    for index in range(data.shape[0]):
        binary_i = [int(c) for c in int2bin(i, col_num)]
        new_columns.append(binary_i)
        if key_dict[data[column].iloc[index]]:
            key_dict[data[column].iloc[index]] = False
            i += 1
    new_columns = np.array(new_columns)
    column_names = [column + "_" + str(i) for i in range(col_num)]
    for i in range(col_num):
        data[column_names[i]] = new_columns[:, i]
    data.drop(column, axis=1, inplace=True)


def gini_score(X, y):
    s = 0
    b = 0
    for y_i in y:
        s += y_i
        b += s
    return 1 - ((2 * b) / (len(X) * s))


print("Script launching")

# ======================== Step 1: Import Data ========================

train = pd.read_csv("input/train_set.csv", nrows=10000)
# train = pd.read_csv("input/train_set.csv")
print("Data loaded")

train_label_one = train[train["Claim_Amount"] != 0]
train_label_zero = train[train["Claim_Amount"] == 0]

# ======================== Step 2: Modify Data ========================

train.replace(to_replace='?', value=np.NaN, inplace=True)

encoding_columns = ["Blind_Submodel", "NVCat"] + ["Cat" + str(i) for i in range(1, 13)]

for col in encoding_columns:
    binary_encoding(col, train)
train.drop(["Blind_Make", "Blind_Model"], axis=1, inplace=True)
print("Binary Encoding over")

train.drop(["Household_ID", "Vehicle", "Row_ID"], axis=1, inplace=True)

# ======================== Step 3: Train Algorithm ========================

train = train.reset_index()
train_y = train["Claim_Amount"]
train_X = train.drop("Claim_Amount", axis=1)

params = {"fit_intercept": [True, False], "normalize": [True, False]}

score = make_scorer(gini_score, greater_is_better=True)
model = GridSearchCV(estimator=LinearRegression(), param_grid=params, scoring=score, cv=KFold(5))
model.fit(X=train_X.as_matrix().astype(np.float), y=train_y.as_matrix().astype(np.float))
print("Model fitted")

