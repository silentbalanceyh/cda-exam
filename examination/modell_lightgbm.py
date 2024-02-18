from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
from sklearn.multiclass import OneVsRestClassifier

from examination.toolkit import *


class ModLLightGBM(Mod):

    def execute(self, executor, classes=1):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        # 分类参数计算
        log_info("（ModLLightGBM）分类参数：%i" % classes)
        model = LGBMClassifier(
            n_estimators=500,
            random_state=0,
            learning_rate=0.002,
            num_leaves=18,
            max_depth=6,
            # eval_metric='mlogloss',
            # num_class=8,
            # objective='multi:softmax'
            # scale_pos_weight = 100,
            # imbalance issue
        )
        model = OneVsRestClassifier(model, n_jobs=-1)
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModLLightGBM）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
