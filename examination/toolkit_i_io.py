import os.path
import pickle
import pandas as pd
import numpy as np  # Not Remove
from gensim.models import Word2Vec

from examination.toolkit_i_logger import *
from examination.toolkit_i_class import *  # Not Remove
from sklearn.datasets import load_svmlight_file

pd.set_option('display.width', 270)
pd.set_option('display.max_columns', None)
PATH_DATA = "data/"
PATH_MODEL = "model/"
PATH_RUNTIME = "runtime/"
ENCODING = "utf-8"

# ========================= 功能函数
'''
输入输出专用
'''


def __in_file(filename=None):
    if filename.endswith("xlsx"):
        return pd.read_excel(filename)
    elif filename.endswith("libsvm"):
        return load_svmlight_file(filename, dtype=np.float64, multilabel=True)
    else:
        return pd.read_csv(filename, encoding=ENCODING, low_memory=False)


def __out_file(df, filename=None):
    if filename.endswith("xlsx"):
        df.to_excel(filename)
    else:
        df.to_csv(filename, encoding=ENCODING, index=False, sep=',')


def in_data(filename=None):
    if filename is None:
        log_error("对不起，请传入 filename 文件名")
    else:
        return __in_file(PATH_DATA + filename)


def in_runtime(filename=None):
    if filename is None:
        log_error("对不起，请传入 filename 文件名")
    else:
        return __in_file(PATH_RUNTIME + filename)


def out_data(df, filename=None):
    if filename is None:
        log_error("对不起，请传入 filename 文件名")
    else:
        __out_file(df, PATH_DATA + filename)


def out_runtime(df, filename=None):
    if filename is None:
        log_error("对不起，请传入 filename 文件名")
    else:
        __out_file(df, PATH_RUNTIME + filename)


def out_model(filename, config):
    with open(PATH_MODEL + filename, 'wb') as file:
        save = config
        pickle.dump(save, file)


def out_w2(model, filename):
    model.save(PATH_MODEL + filename)


def in_model(filename, key="content"):
    if os.path.exists(PATH_MODEL + filename):
        with open(PATH_MODEL + filename, 'rb') as file:
            modelData = pickle.load(file)
            model = modelData[key]
        return model
    else:
        return None


def in_w2(filename):
    return Word2Vec.load(PATH_MODEL + filename)


# ========================= 特定函数
def csv_train(case="actor"):
    return in_runtime(case + "_train.csv")


def csv_test(case="actor"):
    return in_runtime(case + "_test.csv")


def csv_target(case="actor"):
    return in_runtime(case + "_target.csv")


def csv_feature(case="actor"):
    return in_runtime(case + "_train_feature.csv")
