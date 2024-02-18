from examination.toolkit import *

class Actor:
    def __init__(self,
                 f_id,
                 f_target,
                 f_case="actor",
                 f_classes = None):
        self.f_id = f_id
        self.f_target = f_target
        self.f_case = f_case
        self.f_classes = f_classes

        # Delay 延迟执行
        self.splitFn = None
        self.preFn = None
        self.runFn = None
        self.runBFn = None

        # 预测
        self.predictFn = None
        self.scoreFn = None

    def input(self, filename):
        return in_runtime(self.f_case + filename)

    # 拆分数据源
    def fn_split(self,
                 splitFn):
        self.splitFn = splitFn
        return self

    # 执行预处理
    def fn_pre(self,
               preFn):
        self.preFn = preFn
        return self

    # 建模之前
    def fn_run_before(self,
                      runBFn):
        self.runBFn = runBFn
        return self

    # 建模
    def fn_run(self,
               runFn):
        self.runFn = runFn
        return self

    # 预测之前
    def fn_predict(self,
                   predictFn):
        self.predictFn = predictFn
        return self

    # 评分
    def fn_score(self,
                 scoreFn):
        self.scoreFn = scoreFn
        return self

    def execute(self, data_df, phase: RunPhase = None, data2_df=None):
        # 三个基础变量，贯穿全局
        f_id = self.f_id
        f_target = self.f_target
        f_case = self.f_case
        if RunPhase.Split == phase:
            # 「拆分」运行数据拆分
            # data_f 是总数据集
            self.splitFn(data_df, f_case)(f_id, f_target)
            return
        elif RunPhase.Pre == phase:
            # 「特征」读取数据
            # data_f 是训练集 in_runtime(f_case + "_train.csv")
            train_df = data_df
            future = self.preFn(train_df)(f_id, f_target)
            if future is not None:
                out_runtime(future, f_case + "_train_feature.csv")
            return
        elif RunPhase.Model == phase:
            # 「建模-1」
            # future_df = in_runtime(f_case + "_train_feature.csv")
            # test_df = in_runtime(f_case + "_test.csv")
            # data_df -> 训练处理集
            # data2_df -> 测试集
            train_df = data_df

            def prepareFn():
                return self.runBFn(train_df)(f_id, f_target)

            # 「建模-2」
            model = self.runFn()
            if self.f_classes is None:
                _, filename, duration = model.execute(prepareFn)
            else:
                _, filename, duration = model.execute(prepareFn, self.f_classes)
            log_success("----->「第三步」模型训练完成！%s，耗时：%.2f" % (filename, duration))
            return
        elif RunPhase.Predict == phase:
            # 「预测」
            # data_df -> 测试集
            # test_df = in_runtime(f_case + "_test.csv")
            test_df = data_df
            predict_df, filename = self.predictFn(test_df)(f_id, f_target)
            if filename is not None:
                out_runtime(predict_df, filename)
            return
        elif RunPhase.Score == phase:
            # 「评分」
            # expect_df = in_runtime(f_case + "_result.csv")
            # predict_df = in_runtime(f_case + "_predict.csv")
            # data_df -> 期望集
            # data2_df -> 实际预测集
            y_true = data_df
            y_pred = data2_df
            self.scoreFn(y_true, y_pred)(f_id, f_target)
            return
        elif RunPhase.Mix_SP == phase:
            # 「拆分」+ 「特征」
            # 运行数据拆分
            # data_f 是总数据集
            train_df, _, _ = self.splitFn(data_df, f_case)(f_id, f_target)
            # 训练集 in_runtime(f_case + "_train.csv")
            future = self.preFn(train_df)(f_id, f_target)
            out_runtime(future, f_case + "_train_feature.csv")
            return
        elif RunPhase.Mix_MT == phase:
            # 「建模」+ 「预测」
            # 「建模-1」
            train_df = data_df
            test_df = data2_df
            timeStart = time.time()

            def prepareFn():
                return self.runBFn(train_df)(f_id, f_target)

            # 「建模-2」
            model = self.runFn()
            if self.f_classes is None:
                r_model, filename, duration = model.execute(prepareFn)
            else:
                r_model, filename, duration = model.execute(prepareFn, self.f_classes)
            log_success("----->「第三步」模型训练完成！%s，耗时：%.2f" % (filename, duration))
            # 「预测」
            predict_df, filename = self.predictFn(test_df, filename)(f_id, f_target)
            if filename is not None:
                out_runtime(predict_df, filename)
            consume = time.time() - timeStart
            return predict_df, consume

