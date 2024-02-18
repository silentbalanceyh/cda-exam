from examination.toolkit import *
from examination.estimator_wrong import PreWrong
from examination.estimator_outlier import PreOutlier
from examination.estimator_onehot import PreOneHot

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ------------------------- 私有函数


def __cat_normalize(x, outlier=ModeOutlier.Quartile, categorical_features=None, encoder=None):
    if categorical_features is None:
        categorical_features = []
    log_info("步骤1：数据预处理 - Wrong，Outlier")
    pipe = Pipeline([
        ('Wrong_Na', PreWrong(wrong_value=['.', '?'])),
        # XGB 忽略
        ('Outlier', PreOutlier(mode=outlier))
    ])
    # 此处不执行y值
    x = pipe.fit_transform(x)
    # if y is None:
    #     x = pipe.fit_transform(x)
    # else:
    #     x = pipe.fit_transform(x, y)
    # ------------------------- 2.3 数值型编码
    log_info("步骤2：对数值型数据进行编码")
    all_features = x.columns.values.tolist()
    numerical_features = [i for i in all_features if i not in categorical_features]
    # 注意执行顺序，必须是 NA -> Encode
    pipe_cat = Pipeline([
        ("Na", SimpleImputer(strategy='most_frequent')),  # most_frequent, constant fill_value
        # ('encode', OneHotEncoder()), move it out to data step 3
        # One Hot Encoder 的处理结果将会存储成为一个临时模型，该临时模型
        # 会在预测步骤中使用，所以需要将模型存储下来，存储模型之前，还需要针对
        # 所有内容执行 PCA 的动作，所以类别型不能直接放到 Pipeline中。
        ("OneHot", PreOneHot(categorical_features, encoder=encoder))
    ])
    pipe_num = Pipeline([
        ("Na", SimpleImputer(strategy='median')),  # median
        ("Encode", MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', pipe_cat, categorical_features),
        ('num', pipe_num, numerical_features)
    ])
    return preprocessor.fit_transform(x)


# ------------------------- 特征工程

def cat_feature(df_train, f_id, f_target, f_categorical=None, f_outlier=ModeOutlier.Quartile, f_dq=None):
    """
    :param f_dq:
    :param f_outlier:
    :param df_train: 训练数据集
    :param f_id: ID字段
    :param f_target: 目标字段
    :param f_categorical: 特征集专用
    :return: 返回一个DataFrame，可保存成 feature.csv 文件
    """
    df_train = df_train.copy()
    log_matrix("（训练）数据Shape：", df_train.shape)
    log_matrix("（训练）数据目标列：", Counter(df_train[f_target]))
    # --> 结构：ID + Feature + Target
    if f_dq is not None:
        out_runtime(df_train, f_dq)
    df_train.drop(f_id, inplace=True, axis=1)
    # --> 结构：Feature + Target

    # -------------------------------
    df_y = df_train[f_target]
    encoder = LabelEncoder()
    log_message("（Label）二分类处理Y目标列：", df_y.shape)
    df_y = encoder.fit_transform(df_y)
    log_message("（Label）处理后：", df_y.shape)
    # -------------------------------

    log_matrix("（训练）目标列标签化：", df_y)
    df_x = df_train.drop(f_target, axis=1)
    log_matrix("（训练）结构（之前）：", df_x.shape, df_y.shape)
    np_x = __cat_normalize(df_x, categorical_features=f_categorical, outlier=f_outlier)
    log_matrix("（训练）结构（中间）：", np_x.shape, df_y.shape)
    out_df = pd.DataFrame(np_x, dtype=float)
    out_df[f_target] = df_y
    log_matrix("（训练）结构（最终）：", out_df.shape)
    log_success("------>「第二步」特征工程处理完成！", out_df.shape)
    return out_df


def cat_feature_fn(df_train, f_categorical, f_outlier=ModeOutlier.Quartile, f_dq=None):
    return lambda f_id, f_target: cat_feature(df_train, f_id, f_target, f_categorical, f_outlier, f_dq)


# ------------------------- 预测

def cat_predict(df_test, f_id, f_columns, f_categorical=None, f_outlier=ModeOutlier.Quartile):
    """
    :param f_outlier:
    :param df_test: 测试数据
    :param f_id:    目标ID字段
    :param f_columns: 训练集列信息
    :param f_categorical: 特征列
    :return: x_test, y_test
             x_test：测试集X
             y_test：测试集Y
    """
    df_test = df_test.copy()
    log_matrix("（测试）数据Shape：", df_test.shape)
    y_test = df_test[f_id]
    x_test = df_test.drop(f_id, axis=1)
    oneHot = in_model("OneHot.encoder", 'content')
    np_x = __cat_normalize(x_test, categorical_features=f_categorical, encoder=oneHot, outlier=f_outlier)
    log_matrix("（测试）结构（中间）：", np_x.shape, y_test.shape)
    x_test = pd.DataFrame(np_x, dtype=float, columns=f_columns)
    log_matrix("（测试）准备完成：", x_test.shape, y_test.shape)
    return x_test, y_test


def cat_predict_fn(df_test, f_model, f_categorical, o_id, o_target, o_filename=None, f_outlier=ModeOutlier.Quartile):
    # 内部预测函数
    def predict_fn(f_id, f_target=None):
        log_info("--------> 选择模型：\033[31m%s" % f_model)
        # 从模型中读取数据
        mod = in_model(f_model)
        f_columns = in_model(f_model, 'columns')

        x_test, y_test = cat_predict(df_test, f_id, f_columns, f_categorical, f_outlier)
        y_predict = mod.predict(x_test)

        # Y 两次处理（单类和多类）
        np_out = y_combine(y_test, y_predict)
        columns = y_columns(o_id, o_target)
        log_success("----->「第四步」预测结果输出完成！")
        return pd.DataFrame(np_out, columns=columns), o_filename

    return predict_fn
