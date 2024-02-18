from sklearn.base import BaseEstimator, TransformerMixin
from examination.toolkit import *

class PreWrong(BaseEstimator, TransformerMixin):
    def __init__(self,
                 list_num: list = None,  # 数值列
                 list_cat: list = None,  # 类别列
                 wrong_value: list = None,  # 错值列
                 num_limit: int = 8,  # 数值类别相同值判断的阈值
                 ):
        self.list_num = list_num
        self.list_cat = list_cat
        self.wrong_value = wrong_value
        self.num_limit = num_limit

    def fit(self, X, y=None):
        # 列分析：数值列 / 类别列
        self.list_num, self.list_cat = data_column(X, self.list_num, self.list_cat, self.num_limit)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.replace(self.wrong_value, np.nan, inplace=True)
        for col in X.columns:
            _, kind = data_kind(x=X[col], num_limit=self.num_limit)
            if ModeAnalyzer.Numeric == kind:
                X[col] = X[col].astype('float')
            else:
                X[col] = X[col].astype('object')
        log_message("（PreWrong）错误值NA成功处理完成！")
        return X
