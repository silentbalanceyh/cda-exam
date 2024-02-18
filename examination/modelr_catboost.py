from catboost import CatBoostRegressor
from examination.toolkit import *

class ModRCatboost(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # ModRCatboost
        timeStart = time.time()
        """回归
        cb = CatBoostRegressor(
                     n_estimators = 20000, 
                     reg_lambda = 1.0,
                     eval_metric = 'RMSE',
                     random_seed = 42,
                     learning_rate = 0.01,
                     od_type = "Iter",
                     early_stopping_rounds = 2000,
                     depth = 7,
                     cat_features = cate,
                     bagging_temperature = 1.0
        )
        """
        model = CatBoostRegressor(
            n_estimators = 1000,
            reg_lambda = 1.0,
            eval_metric = 'RMSE',
            random_seed = 42,
            learning_rate = 0.01,
            od_type = "Iter",
            early_stopping_rounds = 2000,
            depth = 7,
            bagging_temperature = 1.0
        )
        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=5)
        duration = time.time() - timeStart
        log_info("（ModRCatboost）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration