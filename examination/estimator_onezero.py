import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class PreOneZeroEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoder = LabelEncoder()

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        output = X.copy()
        y_title = self.encoder.fit_transform(self.columns)
        output = pd.DataFrame(output.values[1:,], columns=y_title, index=None)
        return output
