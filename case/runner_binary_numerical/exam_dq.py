# et: Exam Tool（考试专用工具包）
from constant import *
# 原始数据
i_data = ex.in_data("cs-actor_training.csv")
report = ex.DQReport(i_data,V_ID, V_TARGET)
# NReport
dt = report.NDataFrame()
print(dt)
print("-------------------- 对比线 ----------------")
n_data = ex.in_runtime("actor_train_normalized.csv")
report = ex.DQReport(n_data,V_ID, V_TARGET)
dt = report.NDataFrame()
print(dt)
