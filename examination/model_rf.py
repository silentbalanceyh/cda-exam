
from sklearn.ensemble import RandomForestClassifier
from examination.toolkit import *

class ModRForest(Mod):
    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # random forest
        timeStart = time.time()
        model = RandomForestClassifier(
            random_state=0,
            max_depth=18,
            n_estimators=1000,
            oob_score=True,
            n_jobs=32
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModRForest）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
