import xgboost as xgb
from examination.toolkit import *


class ModXGBoost(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        """
        回归
        model_r = xgb.XGBRegressor(
                        max_depth=3,
                        learning_rate=0.1,
                        n_estimators=100,
                        objective='reg:linear', # 此默认参数与 XGBClassifier 不同
                        booster='gbtree',
                        gamma=0,
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        random_state=0
        )
        """
        model = xgb.XGBClassifier(
            n_estimators=1000,
            eta=0.02,
            random_state=0,
            use_label_encoder=False,
            eval_metric='logloss',
            max_depth=6,
            # scale_pos_weight = 100,
            # imbalance issue
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModXGBoost）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
