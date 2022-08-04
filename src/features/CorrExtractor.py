import numpy as np
import pandas as pd

class CorrExtractor(object):
    
    def __init__(self, threshold):
        self.threshold = threshold

    def transform(self, X):
        X_df = pd.DataFrame(X)
        print('Before reducing correlated features: ' + str(X_df.shape[1]))
        corr = X_df.corr()
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if abs(corr.iloc[i,j]) >= self.threshold:
                    if columns[j]:
                        columns[j] = False
        selected_columns = X_df.columns[columns]
        X_corr_filtered = X_df[selected_columns]
        print('After reducing correlated features: ' + str(X_corr_filtered.shape[1]))
        return X_corr_filtered

    def fit(self, X, y=None):
        return self