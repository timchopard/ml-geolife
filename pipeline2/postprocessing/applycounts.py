import numpy as np

def apply_counts(predictions, counts):
    final_output = np.zeros(predictions.shape, dtype=bool)
    
    for idx in range(predictions.shape[0]):
        for col in np.argpartition(
            predictions[idx, :], 
            -counts[idx]
        )[-counts[idx]:]:
            final_output[idx, col] = 1 
    
    return final_output
        