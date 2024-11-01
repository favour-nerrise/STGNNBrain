import numpy as np
import pandas as pd

def functional_connectivity(data, save_pth = None, method='pearson'):
    if method == "pearson":
        connectivity = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            for j in range(i, data.shape[1]):
                connectivity[i, j] = np.corrcoef(data[:, i], data[:, j])[0, 1]
                connectivity[j, i] = connectivity[i, j]
    if save_pth:
        pd.DataFrame(connectivity).to_csv(save_pth)

    return connectivity

