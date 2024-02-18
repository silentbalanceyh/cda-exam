# ------------------------- 私有函数
import pandas as pd

from examination.toolkit import *


def __word_cut(df_train, df_test, f_features=None):
    if f_features is None:
        f_features = []

    # 训练部分
    params = []
    for field in f_features:
        if field is not None:
            params.append(df_train[field])
    # 测试部分
    for field in f_features:
        if field is not None:
            params.append(df_test[field])

    # 多线程执行分词
    counter = len(params)
    pool = ThreadPool(processes=counter)
    result_async = pool.map_async(word_cut, params)
    pool.close()
    pool.join()
    return result_async.get()


def __word_matrix(model, content, title):
    # 生成矩阵向量文件
    log_info("生成向量 ---->")
    timeStart = time.time()
    t_content = word_vectors(content, model)
    log_info("最初尺寸：", len(t_content))
    if title is not None:
        t_title = word_vectors(title, model)
        matrix = np.hstack((np.array(t_content), np.array(t_title)))
    else:
        matrix = np.array((t_content))
    duration = time.time() - timeStart
    log_info("生成矩阵：%.2f 秒，最终尺寸：" % duration, len(matrix))
    return matrix


# ------------------------- 特征工程

def __txt_feature(df_train, df_test, f_title, f_content):
    timeStart = time.time()
    words = __word_cut(df_train, df_test, [f_content, f_title])  # 先content，再title
    duration = time.time() - timeStart
    log_info("分词总共花费 %.2f 秒" % duration)

    # 构造 Word2Vec，根据是否包含 title 划分
    train_title = None
    test_title = None
    timeStart = time.time()
    if f_title is None:
        [train_content, test_content] = words
        train_text = train_content + test_content
    else:
        [train_title, train_content, test_title, test_content] = words
        train_text = train_content + train_title + test_content + test_title
    model = Word2Vec(train_text,
                     vector_size=200,
                     workers=64,
                     min_count=1)
    duration = time.time() - timeStart
    log_info("形成 Word2Vec 模型共花费 %.2f 秒" % duration)
    file_model = "Word2Vec.model"
    log_success("「Finished」训练完成，模型输出文件：%s" % file_model)
    out_w2(model, file_model)

    # X构造
    matrix_train = __word_matrix(model, train_content, train_title)
    matrix_test = __word_matrix(model, test_content, test_title)

    assistData = {
        "m_test": matrix_test,
    }
    file_word = "Word2Vec.word2vec"
    out_model(file_word, assistData)
    log_success("「Finished」辅助数据输出文件：%s" % file_word)

    return pd.DataFrame(matrix_train, dtype=float)

def txt_feature(df_train, df_test, f_id, f_target, f_content, f_title=None, f_encoder=False):
    """
    :param df_train:   训练集
    :param df_test: 测试集
    :param f_id:    ID字段
    :param f_target: 目标字段
    :param f_content: 文本内容
    :param f_title: 文本标题
    :return:
    """
    # 分词计算
    out_df = __txt_feature(df_train, df_test, f_title, f_content)

    # -------------------------------
    df_y = df_train[f_target]
    if f_encoder:
        encoder = in_model("MultiEncoder.encoder", "content")
    else:
        encoder = LabelEncoder()
    log_message("（Label）二分类处理Y目标列：", df_y.shape)
    df_y = encoder.fit_transform(df_y)
    log_message("（Label）处理后：", df_y.shape)
    # -------------------------------

    # df_y = y_l_flat(df_train, f_target)
    out_df[f_target] = df_y
    log_matrix("（训练）结构（最终）：", out_df.shape)
    log_success("------>「第二步」特征工程处理完成！", out_df.shape)
    return out_df


def txt_feature_fn(df_train, df_test, f_content, f_title=None, f_encoder=False):
    return lambda f_id, f_target: txt_feature(df_train, df_test, f_id, f_target, f_content, f_title, f_encoder)

def txt_feature_m(df_train, df_test, f_id, f_target, f_content, f_title=None):
    """
    :param df_train:   训练集
    :param df_test: 测试集
    :param f_id:    ID字段
    :param f_target: 目标字段
    :param f_content: 文本内容
    :param f_title: 文本标题
    :return:
    """
    # 分词计算
    out_df = __txt_feature(df_train, df_test, f_title, f_content)

    # -------------------------------
    df_y = df_train[f_target]
    log_message("（Label）处理后：", df_y.shape)
    # -------------------------------

    # df_y = y_l_flat(df_train, f_target)
    out_df = pd.concat([out_df, df_y], axis=1)
    log_matrix("（训练）结构（最终）：", out_df.shape)
    log_success("------>「第二步」特征工程处理完成！", out_df.shape)
    return out_df

def txt_feature_m_fn(df_train, df_test, f_content, f_title=None):
    return lambda f_id, f_target: txt_feature_m(df_train, df_test, f_id, f_target, f_content, f_title)

# ------------------------- 预测

def txt_predict(df_test, f_id, f_columns):
    """
    :param df_test: 测试数据
    :param f_id:    目标ID字段
    :return: x_test, y_test
             x_test：测试集X
             y_test：测试集Y
    """
    file_word = "Word2Vec.word2vec"
    np_x = in_model(file_word, 'm_test')
    y_test = df_test[f_id]
    log_matrix("（测试）数据Shape：", np_x.shape, y_test.shape)
    x_test = pd.DataFrame(np_x, dtype=float, columns=f_columns)
    log_matrix("（测试）准备完成：", x_test.shape, y_test.shape)
    return x_test, y_test


def txt_predict_fn(df_test, f_model, o_id, o_target, o_filename=None):
    # 内部预测函数
    def predict_fn(f_id, f_target=None):
        log_info("--------> 选择模型：\033[31m%s" % f_model)
        mod = in_model(f_model)
        f_columns = in_model(f_model, 'columns')
        x_test, y_test = txt_predict(df_test, f_id, f_columns)
        y_predict = mod.predict(x_test)
        # Y 两次处理（单类和多类）
        np_out = y_combine(y_test, y_predict)
        columns = y_columns(o_id, o_target)
        log_success("----->「第四步」预测结果输出完成！")
        return pd.DataFrame(np_out, columns=columns), o_filename

    return predict_fn

def txt_predict_m_fn(df_test, f_model, o_id, o_target, o_filename=None):
    # 内部预测函数
    def predict_fn(f_id, f_target=None):
        log_info("--------> 选择模型：\033[31m%s" % f_model)
        mod = in_model(f_model)
        f_columns = in_model(f_model, 'columns')
        x_test, y_test = txt_predict(df_test, f_id, f_columns)
        y_predict = mod.predict(x_test)
        # Y 两次处理（单类和多类）
        encoder = in_model("MultiEncoder.encoder")
        if encoder is not None:
            y_predict = encoder.fit_transform(y_predict)

        np_out = y_combine(y_test, y_predict)
        columns = y_columns(o_id, o_target)

        log_success("----->「第四步」预测结果输出完成！")
        return pd.DataFrame(np_out, columns=columns), o_filename

    return predict_fn