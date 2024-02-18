from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from examination.toolkit import *

class ModDtc(Mod):
    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # dtc
        timeStart = time.time()
        model = DecisionTreeClassifier(
            random_state=0,
            max_depth=18
        )
        # model = GridSearchCV(model, params, cv=5, verbose=1, n_jobs=-1, scoring='f1_micro')
        # model.fit(train_x, train_y)
        model.fit(train_x, train_y)
        # print("Best", model.best_params_)
        duration = time.time() - timeStart
        log_info("（ModDtc）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
