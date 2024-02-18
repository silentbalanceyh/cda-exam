
from lightgbm import LGBMClassifier
from examination.toolkit import *

class ModLightGBM(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # lightBGM
        timeStart = time.time()
        """回归
        model_lgb = lgb.LGBMRegressor(
                            objective='regression',
                            max_depth = 3,
                            learning_rate=0.1, 
                            n_estimators=3938,
                            metric='rmse', 
                            bagging_fraction = 0.8,
                            feature_fraction = 0.8
        )
        """
        model = LGBMClassifier(
            n_estimators=1000,          # 拟合树的数量
            boosting_type='gbdt',       # 设置提升类型
            objective='binary',         # 目标函数
            learning_rate=0.02,         # 学习速率
            metric='logloss',           # 模型度量标准
        )
        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='l1', early_stopping_rounds=5)
        duration = time.time() - timeStart
        log_info("（ModLightGBM）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
