from examination.actor_function import *


class Answer:
    def __init__(self):
        self.report = {}

    def put(self, name, modeler):
        self.report[name] = modeler
        return self

    def build(self, executor, fn_tpl=report_cat_tpl, fn_tpl_add=report_cat_add):
        keys = self.report.keys()
        tpl = fn_tpl(keys)
        for key in keys:
            print("\033[36m ---------> 执行算法：%s --------> \033[37m" % key)
            modeler = self.report[key]
            params = executor(modeler, key + ".csv")
            fn_tpl_add(tpl, params)
        return pd.DataFrame(tpl)

    def run_cat(self, executor):
        data_df = self.build(executor, report_cat_tpl, report_cat_add)
        print("\033[30m")
        print(data_df)

    def run_reg(self, executor):
        data_df = self.build(executor, report_reg_tpl, report_reg_add)
        print("\033[30m")
        print(data_df)

