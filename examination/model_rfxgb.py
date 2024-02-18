from xgboost import XGBRFClassifier
from examination.toolkit import *

class ModRForestXGB(Mod):
    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # random forest xgb
        timeStart = time.time()
        model = XGBRFClassifier(
            random_state=0,
            max_depth=18,
            num_parallel_tree=1000,
            n_jobs=32,
            use_label_encoder=False,
            eval_metric='logloss',
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModRForestXGB）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
