import numpy as np

y_prediction = np.loadtxt("submission.csv", delimiter=",", skiprows=1, usecols=1, dtype=str)
y_actual = np.loadtxt("submission_actual.csv", delimiter=",", skiprows=1, usecols=1, dtype=str)

f1_i_total = 0

for idx, (prediction, actual) in enumerate(zip(y_prediction, y_actual)):
    prediction_set = set(prediction.split())
    actual_set = set(actual.split())
    tp = len(prediction_set.intersection(actual_set))
    fp = len(prediction_set.difference(actual_set))
    fn = len(actual_set.difference(prediction_set))
    f1_i = 1 if fp + fn == 0 else tp / (tp + 0.5 * (fp + fn))
    f1_i_total += f1_i

f_1 = f1_i_total / len(y_actual)
print(f"F1: {f_1:.2f}")