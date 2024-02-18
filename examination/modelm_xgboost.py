
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier

from examination.toolkit import *

class ModMXGBoost(Mod):

    def execute(self, executor, classes=None):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()

        log_info("（ModMXGBoost）分类参数：%i" % classes)
        model = xgb.XGBClassifier(
            n_estimators=1000,
            eta=0.002,
            random_state=0,
            use_label_encoder=False,
            max_depth=6,
            n_jobs=classes,
            eval_metric='logloss'
            # scale_pos_weight = 100,
            # imbalance issue
        )
        model = OneVsRestClassifier(model, n_jobs=-1)
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModMXGBoost）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration