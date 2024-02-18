import math
import threading
import time
from multiprocessing.pool import ThreadPool

import jieba_fast as jieba
from tqdm import tqdm

from examination.toolkit_i_io import *

def __word_vector(cut_words, model, size=200):
    # 新版代码
    word_dict = model.wv.key_to_index
    vector_list = []
    for k in cut_words:
        if k in word_dict:
            index = word_dict[k]
            vector = model.wv.vectors[index]
            vector_list.append(vector)
    if len(vector_list) > 0:
        contentVector = np.array(vector_list).mean(axis=0)
    else:
        contentVector = np.zeros(size)
    return contentVector

def word_cut(list):
    stop_list = []
    cut_list = []
    i = 0
    timeStart = time.time()
    for content in list:
        if isinstance(content, str) and len(content) > 0:
            cutWords = [k for k in jieba.cut(content) if k not in stop_list]
            i += 1
            if i % 1000 == 0:
                log_message("线程 = (%s),处理了%d条，累积花费了%.2f秒" % (
                    threading.currentThread().name, i, time.time() - timeStart))
            cut_list.append(cutWords)
        else:
            cut_list.append([])
    log_info("线程 = (%s) 处理完成，结果：%d" %
             (threading.currentThread().name, len(cut_list)))
    return cut_list


def word_vectors(cut_matrix, model, size=200, threshold=500):
    timeStart = time.time()
    length = len(cut_matrix)
    log_message("词量：%d" % length)

    t_group = math.ceil(length / threshold)
    t_last = length % threshold

    t_params = []
    # 0 -> t_group - 1
    for i in range(0, t_group):
        t_start = 0 + threshold * i
        if i == t_group:
            # 最后一组
            t_end = t_last - 1
        else:
            t_end = threshold * (i + 1)
        t_params.append(cut_matrix[t_start:t_end])

    # 构造执行函数
    def executor(p_list):
        tg_result = []
        tg_name = threading.currentThread().name
        tg_len = len(p_list)
        if tg_len < threshold:
            it_obj = tqdm(range(len(p_list)))
        else:
            it_obj = range(len(p_list))
        for j in it_obj:
            cut_words = p_list[j]
            tg_result.append(__word_vector(cut_words, model, size))
        log_message("当前线程：%s 处理完成，词量：%d" % (tg_name, len(p_list)))
        return tg_result

    # Multi 多线程
    pool = ThreadPool(processes=t_group)
    result_async = pool.map_async(executor, t_params)
    pool.close()
    pool.join()
    log_info("总耗时：%.2f 秒，处理数据：%d" % (time.time() - timeStart, length))
    results = result_async.get()
    text_vector = []
    for item in results:
        text_vector.extend(item)
    return text_vector