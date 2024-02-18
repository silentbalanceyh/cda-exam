from lightgbm import LGBMRegressor

from examination.toolkit import *


class ModRLightGBM(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        model = LGBMRegressor(
            objective='regression',
            max_depth=3,
            learning_rate=0.07,
            n_estimators=10000,
            reg_alpha=0.75,
            reg_lambda=0.45,
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModRLightGBM）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
