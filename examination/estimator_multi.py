from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class PreMultiEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        output = X.copy()
        labelEncoder = LabelEncoder()
        if self.columns is not None:
            for col in self.columns:
                output[col] = labelEncoder.fit_transform(output[col])
        else:
            for column_name, col in output.iteritems():
                output[column_name] = labelEncoder.fit_transform(col)
        return output
