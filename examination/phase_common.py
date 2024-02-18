from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from examination.toolkit import *

# ------------------------- 特征处理（文本和分类只是一个样本均衡问题）
def __xy_adasyn(df_x, df_y):
    ad = ADASYN()
    log_message("ADASYN之前:", Counter(df_y))
    df_x, df_y = ad.fit_resample(df_x, df_y)
    log_message("ADASYN之后:", Counter(df_y))
    return df_x, df_y

def __xy_smote(df_x, df_y):
    sm = SMOTE(random_state=123)
    log_message("SMOTE之前:", Counter(df_y))
    df_x, df_y = sm.fit_resample(df_x, df_y)
    log_message("SMOTE之后:", Counter(df_y))
    return df_x, df_y

def __xy_rate(counter):
    k = counter.most_common(1)
    # 最大的一个
    entry = k[0]            # 不好的玩法
    numerator = entry[1]    # 不好的玩法
    denominator = len(list(counter.elements()))
    rate = numerator / denominator
    rate_req = 0.6
    log_message("均衡率：\033[31m%f\033[37m, 阈值：\033[31m%f\033[37m" % (rate,rate_req))
    return rate > rate_req

def data_modeling(df_feature, f_target):
    """
    :param df_feature: 训练数据
    :param f_target: 目标字段
    :return: x_train, x_test, y_train, y_test, columns
             x_train: 训练集X
             x_test: 测试集X
             y_train: 训练集Y
             y_test：测试集Y
             columns: 训练集列信息
    """
    df_feature = df_feature.copy()
    # 训练数据，--> 结构：Feature + Target
    counter = Counter(df_feature[f_target])
    log_matrix("（训练）数据Shape：", df_feature.shape)
    log_matrix("（训练）数据目标列：", Counter(df_feature[f_target]))
    # Y 值处理
    y_train = df_feature[f_target] ## y_l_flat(df_feature, f_target)
    x_train = df_feature.drop(f_target, axis=1)
    log_matrix("（训练）准备完成：", x_train.shape, y_train.shape)
    columns = x_train.columns

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.18)
    # SMOTE 是否有必要？？？根据 Counter(df_feature[f_target]) 结果来处理
    is_smote = __xy_rate(counter)
    # if is_smote:
        # x_train, y_train = __xy_smote(x_train, y_train)
    log_matrix("（训练）结构（最终）：", x_train.shape, y_train.shape)
    log_matrix("（验证）结构（最终）：", x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test, columns


def data_modeling_fn(df_feature):
    return lambda f_id, f_target: data_modeling(df_feature, f_target)