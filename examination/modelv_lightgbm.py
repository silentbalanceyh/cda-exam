from lightgbm import LGBMClassifier

from examination.toolkit import *


class ModVLightGBM(Mod):

    def execute(self, executor, classes=1):
        train_x, test_x, train_y, test_y, columns = executor()
        # xgboost
        timeStart = time.time()
        # 分类参数计算
        log_info("（ModVLightGBM）分类参数：%i" % classes)
        model = LGBMClassifier(
            n_estimators=2000,
            random_state=0,
            learning_rate=0.002,
            n_jobs=classes,
            objective='multiclass',
            metric='multi_logloss',
            num_leaves=64,
            max_depth=7,
            num_class=classes
            # eval_metric='mlogloss',
            # num_class=8,
            # objective='multi:softmax'
            # scale_pos_weight = 100,
            # imbalance issue
        )
        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=5)
        duration = time.time() - timeStart
        log_info("（ModVLightGBM）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
