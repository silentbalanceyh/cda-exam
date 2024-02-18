from sklearn.pipeline import Pipeline

from examination import PreWrong
from examination.toolkit import *


def reg_feature(df_train, f_id, f_target, f_categorical=None):
    """
    :param df_train:
    :param f_id:
    :param f_target:
    :param f_categorical:
    :return:
    """
    df_train = df_train.copy()
    log_matrix("（训练）数据Shape：", df_train.shape)
    log_matrix("（训练）数据目标列：", Counter(df_train[f_target]))
    # --> 结构：ID + Feature + Target
    df_train.drop(f_id, inplace=True, axis=1)
    df_y = df_train[f_target]
    df_train.drop(f_target, inplace=True, axis=1)

    log_info("步骤1：数据预处理 - Get Dummy")
    f_numeric = [col for col in df_train.columns if col not in f_categorical and col != f_id]
    log_message("数值列（%i）: " % len(f_numeric), f_numeric)
    log_matrix("（训练）结构（之前）：", df_train.shape)
    # log transform skewed numeric features
    df_train[f_numeric] = np.log1p(df_train[f_numeric])
    out_df = pd.get_dummies(df_train, columns=f_categorical)
    log_matrix("（训练）结构（中间）：", out_df.shape)
    out_df[f_target] = df_y
    log_matrix("（训练）结构（最终）：", out_df.shape)
    log_success("------>「第二步」特征工程处理完成！", out_df.shape)
    return out_df


def reg_feature_fn(df_train, f_categorical):
    return lambda f_id, f_target: reg_feature(df_train, f_id, f_target, f_categorical)


def reg_predict(df_test, f_id, f_columns, f_categorical=None):
    """
    :param df_test:
    :param f_id:
    :param f_columns:
    :param f_categorical:
    :return:
    """
    df_test = df_test.copy()
    y_test = df_test[f_id]
    x_test = df_test.drop(f_id, axis=1)
    log_matrix("（测试）数据Shape：", df_test.shape)
    # --> 结构：ID + Feature + Target
    log_info("步骤1：数据预处理 - Get Dummy")
    f_numeric = [col for col in x_test.columns if col not in f_categorical]
    log_message("数值列（%i）: " % len(f_numeric), f_numeric)
    log_matrix("（测试）结构（之前）：", x_test.shape)
    # log transform skewed numeric features
    x_test[f_numeric] = np.log1p(x_test[f_numeric])
    x_test = pd.get_dummies(x_test, columns=f_categorical)
    # Feature shape mismatch, expected: 298, got 292
    x_test = x_test.reindex(columns=f_columns, fill_value=0)

    log_matrix("（测试）结构（最终）：", x_test.shape, y_test.shape)
    return x_test, y_test


def reg_predict_fn(df_test, f_model, f_categorical, o_id, o_target, o_filename=None):
    # 内部预测函数
    def predict_fn(f_id, f_target=None):
        log_info("--------> 选择模型：\033[31m%s" % f_model)
        # 从模型中读取数据
        mod = in_model(f_model)
        f_columns = in_model(f_model, 'columns')

        x_test, y_test = reg_predict(df_test, f_id, f_columns, f_categorical)
        y_predict = mod.predict(x_test)

        # Y 两次处理（单类和多类）
        np_out = y_combine(y_test, y_predict)
        columns = y_columns(o_id, o_target)
        log_success("----->「第四步」预测结果输出完成！")
        return pd.DataFrame(np_out, columns=columns), o_filename

    return predict_fn
