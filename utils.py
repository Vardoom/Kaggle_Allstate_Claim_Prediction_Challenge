import numpy as np

# ======================== Useful functions ========================


def int2bin(i, length):
    return ("{0:{fill}" + str(length) + "b}").format(i, fill='0')


def binary_encoding(col_name, data, binary_length):
    binary_key_dict = dict()
    i = 0
    for value in data[col_name].unique():
        binary_key_dict[value] = int2bin(i, binary_length)
        i += 1
    new_list = np.zeros((data.shape[0], binary_length))
    for i in range(data.shape[0]):
        binary_value = binary_key_dict[data[col_name].iloc[i]]
        for j in range(binary_length):
            new_list[i, j] = binary_value[j]
    for i in range(binary_length):
        data[col_name + "_" + str(i) + "_binenc"] = new_list[:, i]
    data.drop(col_name, axis=1, inplace=True)


def normalize_column(col_name, data):
    minimum = data[col_name].min()
    maximum = data[col_name].max()
    data[col_name + "_norm"] = (data[col_name] - minimum) / (maximum - minimum)
    data.drop(col_name, axis=1, inplace=True)


# ======================== Useful values ========================

max_bit_encoding = 16