from constant import *
data = ex.in_data("train.csv")
report = ex.DQReport(data, V_ID, V_TARGET)
report.SReport()

train_data = ex.in_runtime("actor_train.csv")
print("训练：", len(train_data.columns))
test_data = ex.in_runtime("actor_test.csv")
print("测试：", len(test_data.columns))
feature_data = ex.in_runtime("actor_train_feature.csv")
print("中间：", len(feature_data.columns))