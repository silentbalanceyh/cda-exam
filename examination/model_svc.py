from sklearn.svm import SVC
from examination.toolkit import *

class ModSvc(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # svc
        timeStart = time.time()
        model = SVC(
            # kernel = rbf, linear, poly, sigmoid
            kernel='rbf'
            # class_weight = 'balanced'
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModSvc）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
