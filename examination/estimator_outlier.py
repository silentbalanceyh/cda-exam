from sklearn.base import BaseEstimator, TransformerMixin
from examination.toolkit import *

class PreOutlier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 list_num: list = None,  # 数值列集合
                 mode: ModeOutlier = ModeOutlier.MaxMin,  # 模式设置
                 num_limit: int = 8,  # 数值类别相同值判断的阈值
                 num_min: float = None,  # 极大值处理中的最小值
                 num_max: float = None,  # 极大值处理中的最大值
                 ):
        self.list_num = list_num
        self.num_limit = num_limit
        self.num_min = num_min
        self.num_max = num_max
        self.mode = mode
        log_message("（PreOutlier）当前侦测方法 = %s " % self.mode)

    def fit(self, X, y=None):
        # 列分析：数值列 / 类别列
        self.list_num, _ = data_column(X, self.list_num, None, self.num_limit)
        return self

    def transform(self, X, y=None):
        # 拷贝数据集
        X = X.copy()
        for col in self.list_num:
            describe_ = X[col].describe()
            min_, max_ = 0.0, 0.0
            if ModeOutlier.Quartile == self.mode:
                # 四分位法
                iqr = round(describe_['75%'] - describe_['25%'], 2)
                min_ = round(describe_['25%'] - 1.5 * iqr, 2)
                max_ = round(describe_['75%'] + 1.5 * iqr, 2)
            elif ModeOutlier.MaxMin == self.mode:
                # 极大值/极小值
                min_ = round(X[col].quantile(self.num_min), 2)
                max_ = round(X[col].quantile(self.num_max), 2)
            elif ModeOutlier.Cap == self.mode:
                # 平均值法
                min_ = round(describe_['mean'] - 3 * describe_['std'], 2)
                max_ = round(describe_['mean'] + 3 * describe_['std'], 2)
            elif ModeOutlier.Trust == self.mode:
                # 5% 和 95%
                min_ = round(X[col].quantile(0.05), 2)
                max_ = round(X[col].quantile(0.95), 2)
            else:
                log_warn("不支持的侦测方法！方法 = %s" % self.mode)
            # 修改对应值
            X.loc[X[col] < min_, col] = min_
            X.loc[X[col] > max_, col] = max_
        log_message("（PreOutlier）离群值成功处理完成！")
        return X
