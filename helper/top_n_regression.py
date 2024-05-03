import numpy as np

def get_top_n(data, n):
    topn = np.zeros(data.shape, dtype=bool)
    if type(n) is not int:
        if len(n) != data.shape[0]:
            print(":: [ERROR] n must either be an integer or an array of equal"\
                  " length to the input data")
            return None 
        for idx in range(data.shape[0]):
            true_n = n if type(n) is int else n[idx]
            for col in np.argpartition(data[idx, :], -true_n)[-true_n:]:
                topn[idx, col] = 1
    return topn

