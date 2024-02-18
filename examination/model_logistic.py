
from sklearn.linear_model import LogisticRegression
from examination.toolkit import *

class ModLogistic(Mod):
    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # logistic
        timeStart = time.time()
        model = LogisticRegression(
            random_state=0,
            max_iter=1000,
            n_jobs=32,  # 多线程参数
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModLogistic）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
