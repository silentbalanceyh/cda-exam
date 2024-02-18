import xgboost as xgb
from examination.toolkit import *
import warnings
warnings.filterwarnings('ignore')

class ModRXGBoost(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        model = xgb.XGBRegressor(
            colsample_bytree=0.4,
            gamma=0,
            learning_rate=0.07,
            max_depth=3,
            min_child_weight=1.5,
            n_estimators=10000,
            reg_alpha=0.75,
            reg_lambda=0.45,
            subsample=0.6,
            seed=42
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModRXGBoost）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
