# -----------------------------------------------------------------------------------------------------
#
# 「数据输入区」
#
# -----------------------------------------------------------------------------------------------------
from constant import *
from examination import RunPhase, CaseType
import os
# --------------------- 构造Actor调度器 ---------------------------
# 目录初始化
if not os.path.exists("runtime"):
    os.makedirs("runtime")
if not os.path.exists("model"):
    os.makedirs("model")
runner = ex.Actor(
    V_ID,
    V_TARGET,
    f_classes=CLASSES,
    # f_classes=len(V_TARGET_BINARY) --- 多分类时使用
    # p_case="actor" --- 统一文件名专用案例前缀
)
# ---------------------------------------------------------------

# 数据部分
# Data1：原始训练文件
# Data2：训练集（actor_train.csv）
# Data3：验证集（actor_test.csv）
# Data4：验证目标集（actor_target.csv）
# Data5：预处理特征集（actor_train_feature.csv）
# Data6：验证预测结果

# --------------------- Phase 0：文件格式处理 ----------------------
"""
Phase 0「特殊步骤」

原始格式
    ID,     message,    a1,     a2,     a3
    id1,    msg1,       1,      0,      0
    id2,    msg2,       0,      1,      0
    id3,    msg3,       0,      0,      1

新格式
    ID,     message,    Target
    id1,    msg1,       a1
    id2,    msg2,       a2
    id3,    msg3,       a3

    多分类文件格式最终可全部转换成"一列多值"的核心格式，即类似上述新格式结构。
"""


def run_revert():
    # 原始数据源文件
    # 输入：
    # /data/training.xlsx
    df_binary = ex.in_data(IN_PRE)  # Data0 ----------------------------->
    df_out = ex.y_revert(
        df_binary,  # 原始训练集
        f_source=V_TARGETS,  # 原始属性名，Array结构
        f_target=V_TARGET  # 目标属性名，String结构
    )
    ex.log_matrix("还原过后的数据：", df_out.shape)

    # 生成新数据源文件
    # 输出：
    # /data/training_data.csv
    ex.out_data(df_out, IN_SOURCE)  # ----------------------------> Data1
    ex.log_success("「00」格式转换完成！")


# --------------------- Phase 1：拆分数据集 ----------------------
"""
RunPhase = Split
Phase 1，拆分训练数据集

根据 p_radio 进行比例拆分：
    原始训练集 ----   训练集（ID + Features + Target）         0.8
                    验证集（ID + Features）                  0.2
                    目标集（ID +            Target）         0.2
                
核心调用API：
    exam.data_split_fn
    exam.data_split

「核心代码」：

    1. 拆分核心逻辑
    
    --------------------------------------------------------------------------------
    （1）拆分数据集
        train_df, test_df = train_test_split(df_data, test_size=p_radio)
        
    （2）合并索引
        index = f_id + f_target ( V_ID + V_TARGET )
        
        （2.1）可选步骤（多分类使用）
            生成 MultiEncoder.encoder（后续步骤会使用）
            /model/MultiEncoder.encoder
        
    （3）去掉对应列
        target_df = test_df[index]      # 只包含 ID 和 Target
        test_df.drop(f_target, axis=1)  # 只包含 ID 和 Features
    --------------------------------------------------------------------------------
    
    生成三份文件，前缀使用 p_case 的值
    /runtime/actor_train.csv
    /runtime/actor_test.csv
    /runtime/actor_target.csv
"""


def run_splitting():
    # 新数据源文件
    # 输入：
    # /data/training_data.csv
    df_data = ex.in_data(IN_SOURCE)  # Data1 -------------------------->
    runner.fn_split(
        # 拆分数据集专用函数
        #   data_df - 输入数据集
        #   p_case - 当前案例名称，默认 actor 前缀
        lambda data_df, p_case: ex.data_split_fn(
            df_data=data_df,  # 原始训练集
            p_case=p_case,  # 对应Case名
            p_radio=0.2,  # 训练和验证集比例
            f_target_binary=V_TARGETS  # （多分类）专用
        )
    )

    # 输出（三份）
    # /runtime/actor_train.csv
    # /runtime/actor_test.csv
    # /runtime/actor_target.csv
    runner.execute(df_data, RunPhase.Split)  # -----------------Data2, Data3, Data4


# --------------------- Phase 2：特征提取 （分类/文本/数值）----------------------
"""
RunPhase = Pre
Phase 2，特征提取

核心调用API：
    exam.cat_feature_fn             - 分类
    exam.cat_feature                - 分类
    exam.txt_feature_fn             - 文本挖掘
    exam.txt_feature                - 文本挖掘

「核心代码」：
    
    1. 分类部分（CaseType.Binary）
    
    --------------------------------------------------------------------------------
    （1）训练集移除ID列                  
            ID + Features + Target              ===>        Features + Target
        df_train.drop(f_id, inplace=True, axis=1)
        
    （2）执行Y的标签化
        y值中的LabelEncoder流程
        df_y = df_train[f_target]
        
    （3）训练集X移除Target列             
            Features + Target                   ===>        Features
        df_x = df_train.drop(f_target, axis=1)
        
    （4）Y和X执行预处理
        1）空值补充
        2）错误值补充
        3）类型和数值分开处理
            3.1）类型：
                1 - 最常出现值填充（取频率最高的）
                2 - 对类型字段中的值执行独热编码      「自定义流程，还包括编码器加载和PCA处理」
            3.2）数值：
                1 - 均值填充
                2 - 极值MinMax正规化
        
    （5）合并最终DataFrame
        df_x[f_target] = df_y
            Features                            ===>        Features + Target
    --------------------------------------------------------------------------------
    
    2. 文本部分（CaseType.Textual）
    
    --------------------------------------------------------------------------------
    （1）根据标题和内容分词
        
        content 必须，title 可选，同时处理训练集和验证集
        多线程分词（四个线程）
        words = __word_cut(df_train, df_test, [f_content, f_title])
        
    （2）构造Word2Vec模型
        2.1）不传title
            train_text = train_content + test_content
        2.2）传入title
            train_text = train_content + train_title + test_content + test_title
        model = Word2Vec(train_text,
                     vector_size=200,
                     workers=64,
                     min_count=1)
    
    （3）生成模型文件
        /model/Word2Vec.model
        /model/Word2Vec.model.syn1neg.npy
        /model/Word2Vec.model.wv.vectors.npy
    
    （4）构造向量矩阵
        训练的X
        matrix_train = __word_matrix(model, train_content, train_title)
        测试的X
        matrix_test = __word_matrix(model, test_content, test_title)
    
    （5）存储测试的X
        生成辅助文件 
        /model/Word2Vec.word2vec
        
    （6）执行Y的标签化
        y值中的LabelEncoder流程
        df_y = df_train[f_target]
    
    （7）合并最终DataFrame
        df_x[f_target] = df_y
            Features                            ===>        Features + Target
    --------------------------------------------------------------------------------
    
    生成一份文件，前缀使用 p_case 的值
    /runtime/actor_train_feature.csv
"""


def run_feature():
    # 训练数据源，测试数据源
    # 输入：
    # /runtime/actor_train.csv
    # /runtime/actor_csv.csv
    i_train = ex.csv_train()  # Data2 ----------------------------->
    if CaseType.Textual == CASE:
        i_test = ex.csv_test()  # Data3 ----------------------------->
        runner.fn_pre(
            # 特征工程
            #   df_train: 训练集
            lambda df_train: ex.txt_feature_fn(
                df_train=df_train,  # 训练集
                df_test=i_test,  # 验证集（生成文本测试专用向量矩阵）
                # f_title=V_TITLE,      # 包含标题
                f_content=F_CONTENT  # 文本内容
            )
        )
    elif CaseType.Binary == CASE:
        runner.fn_pre(
            # 特征工程
            #   df_train: 训练集
            lambda df_train: ex.cat_feature_fn(
                df_train=df_train,
                f_categorical=F_FEATURES  # 分类中影响结果的特征集
            )
        )
    elif CaseType.Numeric == CASE:
        runner.fn_pre(
            # 特征工程
            #   df_train: 训练集
            lambda df_train: ex.cat_feature_fn(
                df_train=df_train,
                f_categorical=F_FEATURES, # 分类中影响结果的特征集
                f_outlier=ex.ModeOutlier.Trust,
                f_dq="actor_train_normalized.csv"
            )
        )
    elif CaseType.TextualMulti == CASE:
        i_test = ex.csv_test()  # Data3 ----------------------------->
        runner.fn_pre(
            # 特征工程
            # df_train: 训练集
            lambda df_train: ex.txt_feature_m_fn(
                df_train=df_train,
                df_test=i_test,        # 此处需要测试集同时执行分词操作
                f_content=F_CONTENT
            )
        )
    elif CaseType.Regression == CASE:
        runner.fn_pre(
            # 特征工程
            #   df_train: 训练集
            lambda df_train: ex.reg_feature_fn(
                df_train=df_train,
                f_categorical=F_FEATURES  # 分类中影响结果的特征集
            )
        )
    # 输出
    # /runtime/actor_train_feature.csv
    runner.execute(i_train, RunPhase.Pre)  # ----------------------------> Data5


# --------------------- Phase 3：训练模型----------------------
"""
RunPhase = Model
Phase 3，训练模型

核心调用API：
    exam.data_modeling_fn
    exam.data_modeling

核心算法表
    1）二分类
        exam.ModLogistic                        逻辑回归
        exam.ModDtc                             决策树
        exam.ModLightGBM                        LightGBM
        exam.ModCatboost                        CatBoost
        exam.ModXGBoost                         XGBoost
        exam.ModMLP                             多层感知机
        exam.ModSvc                             支持向量机
        exam.ModRForest                         随机森林
        exam.ModRForestXGB                      随机森林 + XGBoost
    2）多分类（单列多值）
        exam.ModVLightGBM                       LightGBM
        exam.ModVCatboost                       Catboost
        exam.ModVXGBoost                        XGBoost
        exam.ModVRForest                        随机森林
        exam.ModVRForestXGB                     随机森林 + XGBoost

「核心代码」：

    1. 建模之前处理数据
    --------------------------------------------------------------------------------
    （1）拆分特征文件
        原始文件：
            Data5 = Features + Target
            X     = Features
            Y     =            Target             
        
            y_train = df_feature[f_target] 
            x_train = df_feature.drop(f_target, axis=1)
    （2）拆分准备样本
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
        
    （3）是否执行样本均衡
        SMOTE模式
        ADASYN模式
    
    最终返回 x_train, x_test, y_train, y_test, columns
    --------------------------------------------------------------------------------
    
    生成模型文件
    /model/XXX.model
"""


def run_model():
    # 输入：
    # /runtime/actor_train_feature.csv
    i_feature = ex.csv_feature()  # Data5 ----------------------------->
    runner.fn_run_before(ex.data_modeling_fn) \
        .fn_run(MODELER)
    # 输出
    # /model/XXX.model（模型文件）
    runner.execute(i_feature, RunPhase.Model)


# --------------------- Phase 4：预测 --------------------------------
"""
RunPhase = Predict
Phase 4，预测数据集

核心调用API：
    exam.cat_predict_fn                     -- 分类
    exam.cat_predict                        -- 分类
    exam.txt_predict_fn                     -- 文本挖掘
    exam.txt_predict                        -- 文本挖掘

「核心代码」：

    1. 预测之前处理数据
    --------------------------------------------------------------------------------
    （1）加载模型
        从模型文件 model/XXX.model 中加载模型
        mod = in_model(f_model)
    
    （2）读取存储的列
        f_columns = in_model(f_model, 'columns')
    
    （3）生成测试专用文件
        步骤同训练集（略）
        x_test, y_test = cat_predict(df_test, f_id, f_columns, f_categorical)
    
    （4）预测
        y_predict = mod.predict(x_test)
    
    （5）合并结果集
        np_out = y_combine(y_test, y_predict)
        columns = y_columns(o_id, o_target)
        return pd.DataFrame(np_out, columns=columns), o_filename
    --------------------------------------------------------------------------------
    
    生成验证结果文件
    /runtime/Mod.csv
"""


def run_predict():
    # 输入：
    # /runtime/actor_test.csv
    i_test = ex.csv_test()  # Data3 ----------------------------->
    if CaseType.Textual == CASE:
        runner.fn_predict(
            lambda df_test: ex.txt_predict_fn(
                df_test=df_test,
                f_model=OUT_MODEL,
                o_id=O_ID,
                o_target=O_TARGET,
                o_filename=OUT_RESULT
            )
        )
    elif CaseType.Binary == CASE:
        runner.fn_predict(
            lambda df_test: ex.cat_predict_fn(
                df_test=df_test,
                f_model=OUT_MODEL,
                f_categorical=F_FEATURES,
                o_id=O_ID,
                o_target=O_TARGET,
                o_filename=OUT_RESULT
            )
        )
    elif CaseType.Numeric == CASE:
        runner.fn_predict(
            lambda df_test: ex.cat_predict_fn(
                df_test=df_test,
                f_model=OUT_MODEL,
                f_categorical=F_FEATURES,
                o_id=O_ID,
                o_target=O_TARGET,
                o_filename=OUT_RESULT,
                f_outlier=ex.ModeOutlier.Trust
            )
        )
    elif CaseType.TextualMulti == CASE:
        runner.fn_predict(
            lambda df_test: ex.txt_predict_m_fn(
                df_test=df_test,
                f_model=OUT_MODEL,
                o_id=O_ID,
                o_target=O_TARGET,
                o_filename=OUT_RESULT
            )
        )
    elif CaseType.Regression == CASE:
        runner.fn_predict(
            lambda df_test: ex.reg_predict_fn(
                df_test=df_test,
                f_model=OUT_MODEL,
                f_categorical=F_FEATURES,
                o_id=O_ID,
                o_target=O_TARGET,
                o_filename=OUT_RESULT
            )
        )
    # 输出
    # /runtime/Mod.csv（根据选择模型名称而定）
    return runner.execute(i_test, RunPhase.Predict)


# --------------------- Phase 5：评分 ----------------------------
"""
RunPhase = Score
Phase 5, 评分

核心调用API：
    exam.data_score_fn
    exam.data_score

「核心代码」（略）
"""


def run_score():
    # 输入：
    # /runtime/actor_target.csv
    i_true = ex.csv_target()  # Data4 ----------------------------->
    i_pred = ex.in_runtime(OUT_RESULT)
    if CaseType.Regression == CASE:
        runner.fn_score(
            lambda df_true, df_pred: ex.reg_score_fn(
                df_true=df_true,
                df_predict=df_pred,
                o_target=O_TARGET
            )
        )
    else:
        runner.fn_score(
            lambda df_true, df_pred: ex.cat_score_fn(
                df_true=df_true,
                df_predict=df_pred,
                o_target=O_TARGET
            )
        )
    return runner.execute(i_true, RunPhase.Score, i_pred)

# -----------------------------------------------------------------------------------------------------
#
# 「混合逻辑区域」
#
# -----------------------------------------------------------------------------------------------------
#  组合流程
#       常用1：预测 + 评分
def mix_modeling(f_modeler, f_out=None):
    if CaseType.Textual == CASE:
        return ex.report_txt(
            modeler=f_modeler,
            f_id=V_ID,
            f_target=V_TARGET,
            o_filename=f_out,
            o_id=O_ID,
            o_target=O_TARGET
        )
    elif CaseType.Binary == CASE:
        return ex.report_cat(
            modeler=f_modeler,
            f_id=V_ID,
            f_target=V_TARGET,
            f_features=F_FEATURES,
            o_filename=f_out,
            o_id=O_ID,
            o_target=O_TARGET
        )
    elif CaseType.Numeric == CASE:
        return ex.report_cat(
            modeler=f_modeler,
            f_id=V_ID,
            f_target=V_TARGET,
            f_features=F_FEATURES,
            o_filename=f_out,
            o_id=O_ID,
            o_target=O_TARGET,
            f_outlier=ex.ModeOutlier.Trust
        )
    elif CaseType.Regression == CASE:
        return ex.report_reg(
            modeler=f_modeler,
            f_id=V_ID,
            f_target=V_TARGET,
            f_features=F_FEATURES,
            o_filename=f_out,
            o_id=O_ID,
            o_target=O_TARGET
        )


#  组合流程
#       常用3：最终考试
def run_exam():
    # 做一次文件转换
    # 原训练集 -> runtime/actor_train.csv
    # 原测试集 -> runtime/actor_test.csv
    run_feature()
    run_model()
    run_predict()

