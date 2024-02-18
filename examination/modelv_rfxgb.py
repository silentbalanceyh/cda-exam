
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier

from examination.toolkit import *

class ModVRForestXGB(Mod):

    def execute(self, executor, classes=None):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()

        log_info("（ModVRForestXGB）分类参数：%i" % classes)
        model = XGBRFClassifier(
            random_state=0,
            max_depth=18,
            num_parallel_tree=1000,
            n_jobs=32,
            use_label_encoder=False,
            objective='multi:softprob',
            eval_metric='mlogloss',
            eta=0.02
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModVRForestXGB）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration