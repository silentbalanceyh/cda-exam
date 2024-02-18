from sklearn.preprocessing import MultiLabelBinarizer
from examination.toolkit_i_io import *


def y_columns(f_id, f_target):
    """
    :param f_id:    ID字段名
    :param f_target:目标字段名，可以 str，也可以 []
    :return:
    """
    indexes = []
    indexes.append(f_id)
    if isinstance(f_target, str):
        indexes.append(f_target)
    else:
        indexes = indexes + f_target
    return indexes


def y_combine(y_test, y_predict):
    left = y_test.values
    right = y_predict
    log_matrix("结果集：", left.shape, right.shape)
    dm_left = left.ndim
    dm_right = right.ndim
    if dm_left == dm_right:
        log_info("单分类结果合并 ---->")
        # 单类型专用
        return np.vstack((left, right)).T
    else:
        log_info("多分类结果合并 ---->")
        # 多类型专用
        return np.hstack((left.reshape(-1, 1), right))


def y_revert(df_binary, f_source, f_target):
    encoder = MultiLabelBinarizer(classes=f_source)
    encoder.fit(f_source)

    df_y = df_binary[f_source]
    df_y_pre = encoder.inverse_transform(df_y.values)
    np_y = []
    for item in df_y_pre:
        np_y.append(item[0])
    df_out = df_binary.drop(f_source, axis=1)
    df_out[f_target] = np_y
    return df_out

def y_transform(df_binary, f_source, f_target):
    encoder = MultiLabelBinarizer(classes=f_source)
    df_y = df_binary[f_target]
    # 由于是多标签，所以需要使用这种方式来处理最终结果，记住此处是二维数组
    df_y_m = []
    for item in df_y:
        df_y_m.append([item])
    df_y_post = encoder.fit_transform(df_y_m)
    df_binary = df_binary.copy()
    df_binary = df_binary.drop(f_target, axis=1)
    df_binary[f_source] = df_y_post
    return df_binary

def y_result_1(df_binary, f_target):
    df_binary = df_binary.copy()
    encoder = in_model("MultiEncoder.encoder")
    df_binary[f_target] = encoder.inverse_transform(df_binary[f_target])
    return df_binary
