import pandas as pd
import numpy as np

from utils import *


def main():

    # ======================== Step 1 : Load Data ========================

    file_name_train = "train_set"
    file_name_test = "test_set"
    extension = ".csv"

    data = pd.read_csv("input/" + file_name_train + extension, nrows=100000)
    train_split = data.shape[0]
    test = pd.read_csv("input/" + file_name_test + extension, nrows=5000)
    data = data.append(test)
    print("File loaded")

    # ======================== Step 2: Process Data ========================

    data.drop(["Row_ID", "Household_ID", "Vehicle", "Calendar_Year", "Blind_Make", "Blind_Model"], axis=1, inplace=True)

    col_names = data.columns

    has_label = False
    if "Claim_Amount" in col_names:
        has_label = True
        data_y = data["Claim_Amount"]
        data.drop("Claim_Amount", axis=1, inplace=True)

    col_names = data.columns

    ord_columns = list()
    cat_columns = list()
    for col in col_names:
        ty = data.dtypes
        if ty[col] in [np.dtype("int64"), np.dtype("float64")]:
            ord_columns.append(col)
        else:
            cat_columns.append(col)

    for col_name in cat_columns:
        print("Binary encoding for column: ".format(col_name))
        binary_encoding(col_name, data, max_bit_encoding)
    print("Binary Encoding over")

    for col_name in ord_columns:
        print("Normalizing for column: ".format(col_name))
        normalize_column(col_name, data)
    print("Normalizing over")

    col_names = data.columns
    i = 0
    j = 0
    for col_name in col_names:
        print("Renaming column")
        if "_binenc" in col_name:
            data.rename({col_name: "cat_binenc_" + str(i)}, inplace=True, axis=1)
            i += 1
        if "_norm" in col_name:
            data.rename({col_name: "var_norm_" + str(j)}, inplace=True, axis=1)
            j += 1

    if has_label:
        data["label_ord"] = data_y
        data["label_cat"] = pd.Series(data_y != 0, dtype=np.dtype("int64"))

    print("Processing over")

    # ======================== Step 3: Save Data ========================

    with open("input/" + file_name_train + "_data.npy",  mode='wb') as numpy_file:
        np.save(numpy_file, data.iloc[:train_split, :])
    with open("input/" + file_name_test + "_data.npy",  mode='wb') as numpy_file:
        np.save(numpy_file, data.iloc[train_split:, :])
    with open("input/names.npy",  mode='wb') as numpy_file:
        np.save(numpy_file, data.columns)
    print("File saved in numpy format")


if __name__ == "__main__":
    main()
