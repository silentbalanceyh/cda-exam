
from sklearn.neural_network import MLPClassifier
from examination.toolkit import *

class ModMLP(Mod):

    def execute(self, executor):
        train_x, test_x, train_y, test_y, columns = executor()
        # mlp
        timeStart = time.time()
        model = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            alpha=0.001,
            max_iter=1000,
            random_state=0,
            early_stopping=True
        )
        model.fit(train_x, train_y)
        duration = time.time() - timeStart
        log_info("（ModMLP）分数：%s, 耗时 %.2f" % (model.score(test_x, test_y), duration))
        # 保存当前模型（固定文件名）
        filename = self.__class__.__name__ + ".model"
        out_model(filename, {
            'content': model,
            'columns': columns
        })
        return model, filename, duration
