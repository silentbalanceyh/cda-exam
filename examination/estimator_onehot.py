
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from examination.toolkit import *

class PreOneHot(BaseEstimator, TransformerMixin):
    def __init__(self,
                 list_cat: list = None,
                 encoder=None):
        self.list_cat = list_cat
        self.encoder = encoder
        if encoder is None:
            self.out = True
        else:
            self.out = False
        self.columns = []

    def fit(self, X, Y=None):
        log_info("步骤3：（PreOneHot）类别型编码")
        if self.encoder is None:
            enc = OneHotEncoder(handle_unknown='ignore')
        else:
            enc = self.encoder
        categorical_cols = X[:, 0:len(self.list_cat)]
        enc.fit(categorical_cols)
        self.columns = categorical_cols
        self.executor = enc
        return self

    def transform(self, X, Y=None):
        oneHot = self.executor.transform(self.columns).A
        log_info("步骤4：（PreOneHot）类别型执行 PCA")

        # PCA分析
        pca = PCA(n_components=0.95)
        pca.fit(oneHot)
        log_message("（PreOneHot）PCA结果：n_components_", pca.n_components_)

        pca_model = PCA(n_components=pca.n_components_)
        oneHot = pca_model.fit_transform(oneHot)
        log_message("（PreOneHot）执行完成PCA信息：", pca_model.n_components_)

        # 组合计算最终结果
        X = X[:, len(self.columns):-1]
        X = np.hstack((oneHot, X))

        if self.out:
            log_message("（PreOneHot）模型输出文件：", "OneHot.encoder")
            out_model("OneHot.encoder", {
                'content': self.executor
            })
        return X
