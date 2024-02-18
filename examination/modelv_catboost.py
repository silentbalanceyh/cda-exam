from catboost import CatBoostClassifier
from examination.toolkit import *

class ModVCatboost(Mod):

    def execute(self, executor, classes=None):
        train_x, test_x, train_y, test_y, columns = executor()
        # lightBGM
        timeStart = time.time()
        log_info("（ModVCatboost）分类参数：%i" % classes)
        model = CatBoostClassifier(
            learning_rate=0.05,
            loss_function='MultiClass',
            classes_count=classes
        )
        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=5)
        duration = time.time() - timeStart
        log_info("（ModVCatboost）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration