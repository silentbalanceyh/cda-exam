from collections import Counter
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, explained_variance_score, \
    mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from examination.toolkit_o_y import *
from examination.estimator_onezero import PreOneZeroEncoder


def __file_out(flag, p_case): return PATH_RUNTIME + (
    "%s.csv" % flag if p_case is None else "%s_%s.csv" % (p_case, flag))


# ------------------------- 数据集处理

def data_split(df_data, f_id, f_target, p_case=None, p_radio=0.16, f_target_binary=None):
    """
    :param df_data: 全数据集
    :param f_id:    ID字段名
    :param f_target:目标字段名
    :param f_target_binary: 原始
    :param p_case:   「Optional」文件前缀
    :param p_radio:  「Optional」拆分数据集的比例
    :return:  train, test, target
              train：训练集
              test：测试集
              target: 测试集的Y值
    """
    log_info("导入数据Shape：", df_data.shape)
    train_df, test_df = train_test_split(df_data, test_size=p_radio)
    indexes = y_columns(f_id, f_target)
    target_df = test_df[indexes]  # 测试集Y值

    if f_target_binary is not None:
        target_df = target_df.copy()
        if isinstance(f_target, str):
            encoder = LabelEncoder()
            target_df[f_target] = encoder.fit_transform(target_df[f_target])
            out_model("MultiEncoder.encoder", {'content': encoder})

    file_train = __file_out("train", p_case)
    log_message("训练数据集：%s" % file_train)
    log_message("训练集Y：", dict(Counter(train_df[indexes])))
    train_df.to_csv(file_train, index=False)

    file_test = __file_out("test", p_case)
    log_message("验证数据集：%s" % file_test)
    log_message("验证集Y：", dict(Counter(test_df[indexes])))
    test_df = test_df.drop(f_target, axis=1)  # 验证集中必须去掉 Target
    test_df.to_csv(file_test, index=False)

    file_target = __file_out("target", p_case)
    log_message("验证结果：%s" % file_target)
    target_df.to_csv(file_target, index=False)

    log_info("训练数据Shape：", train_df.shape)
    log_info("验证数据Shape：", test_df.shape)
    log_success("------>「第一步」数据拆分完成！比例：\033[31m%.2f" % p_radio)
    return train_df, test_df, target_df


def data_split_fn(df_data, p_case, p_radio=0.2, f_target_binary=None):
    return lambda f_id, f_target: data_split(df_data, f_id, f_target, p_case, p_radio, f_target_binary)


# ------------------------- 评分和报表

def cat_score(df_true, df_predict, f_target, o_target):
    # ValueError: Classification metrics can't handle a mix of binary and continuous targets
    # 此处可以 astype("int") 或 astype("float") 来解决
    y_true = df_true[f_target].astype("float")
    y_pred = df_predict[o_target].astype("float")
    f1 = f1_score(y_true, y_pred, average=None)
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    # 精确率
    precision = precision_score(y_true, y_pred, average=None)
    precision_micro = precision_score(y_true, y_pred, average="micro")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    # 召回率
    recall = recall_score(y_true, y_pred, average=None)
    recall_micro = recall_score(y_true, y_pred, average="micro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    log_info("F1宏观 Macro：\033[31m", macro)
    log_info("F1微观 Micro：\033[31m", micro)
    log_info("准确率 Accuracy：\033[31m", accuracy)
    log_info("精确率宏观 Precision Macro：\033[31m", precision_macro)
    log_info("精确率微观 Precision Micro：\033[31m", precision_micro)
    log_info("召回率宏观 Macro：\033[31m", recall_macro)
    log_info("召回率微观 Micro：\033[31m", recall_micro)
    return {
        'f1_macro': macro,
        'f1_micro': micro,
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro
    }

"""
回归评分
1. 可解释方差值
from sklearn.metrics import explained_variance_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred) 

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
explained_variance_score(y_true, y_pred, multioutput='raw_values')
explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7])

2. 平均绝对误差
from sklearn.metrics import mean_absolute_error
3. 均方误差
from sklearn.metrics import mean_squared_error
4. 中值绝对误差
from sklearn.metrics import median_absolute_error
5. 决定系数
from sklearn.metrics import r2_score

r2_score(y_true, y_pred) 
r2_score(y_true, y_pred, multioutput='variance_weighted')
r2_score(y_true, y_pred, multioutput='uniform_average')
r2_score(y_true, y_pred, multioutput='raw_values')
r2_score(y_true, y_pred, multioutput=[0.3, 0.7])
"""

def reg_score(df_true, df_predict, f_target, o_target):
    # ValueError: Classification metrics can't handle a mix of binary and continuous targets
    # 此处可以 astype("int") 或 astype("float") 来解决
    y_true = df_true[f_target].astype("float")
    y_pred = df_predict[o_target].astype("float")
    explained = explained_variance_score(y_true, y_pred)
    absolute = mean_absolute_error(y_true, y_pred)
    squared = mean_squared_error(y_true, y_pred)
    median = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    log_info("可解释方差：\033[31m", explained)
    log_info("平均绝对误差：\033[31m", absolute)
    log_info("均方误差：\033[31m", squared)
    log_info("中值绝对误差：\033[31m", median)
    log_info("决定系数R2：\033[31m", r2)

    return {
        'explained': explained,
        'absolute': absolute,
        'squared': squared,
        'median': median,
        'r2': r2
    }

def cat_score_fn(df_true, df_predict, o_target):
    return lambda f_id, f_target: cat_score(df_true, df_predict, f_target, o_target)

def reg_score_fn(df_true, df_predict, o_target):
    return lambda f_id, f_target: reg_score(df_true, df_predict, f_target, o_target)