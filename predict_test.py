import pandas as pd
import pickle
import numpy as np

# ======================== Step 1 : Load Data ========================

model_name = "SVM_0"

with open("models/" + model_name + ".pkl", "rb") as model_file:
    model = pickle.load(model_file)

file_name = "test_set"

with open("input/" + file_name + "_data.npy", "rb") as np_file:
    data = np.load(np_file)
with open("input/names.npy", "rb") as np_file:
    col_names = np.load(np_file)

test_data = pd.DataFrame(data, columns=col_names)

test_y = np.array(test_data["label_cat"], dtype=int)
test_x = test_data.drop(["label_cat", "label_ord"], axis=1)

print("Data loaded")

# ======================== Step 2 : Predict Result Probabilities ========================

test_predict = model.predict(test_x)
proba = model.predict_proba(test_x)

