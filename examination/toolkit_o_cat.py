from examination.toolkit_i_io import *

def data_kind(x: pd.Series, num_limit: int = 8):
    """
    :param x:         数据集
    :param num_limit: 多少重复值判断为类别列
    :return:          返回当前列是 Numeric,Categorical
    """
    x = x.astype('str')
    x = x.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    x.dropna(inplace=True)
    if x.nunique() > num_limit:
        kind = ModeAnalyzer.Numeric
    else:
        kind = ModeAnalyzer.Categorical
    return x, kind


def data_column(X, list_num=None, list_cat=None, num_limit: int = 8):
    """
    :param X:           数据集
    :param list_num:    数值列
    :param list_cat:    类别列
    :param num_limit:   限制多少重复值为类别列
    :return:
    """
    if list_cat is None:
        list_cat = []
    if list_num is None:
        list_num = []
    # 拷贝数据集
    X = X.copy()
    # 读取引用
    for col in X.columns:
        _, kind = data_kind(x=X[col], num_limit=num_limit)
        if ModeAnalyzer.Numeric == kind:
            list_num.append(col)
        elif ModeAnalyzer.Categorical == kind:
            list_cat.append(col)
    return list_num, list_cat
# 将多列合并成一列，追加列名到值上
def data_flat(df_data, f_target):
    df_target = df_data[f_target]
    print(df_target)
    pass
