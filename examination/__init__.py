"""
CDA考试专用工具包

常量

    V_CASE                  - 当前执行案例（全程绑定，可运行不同的案子）
    V_ID                    - 目标ID
    V_TARGET                - 目标值
    V_TITLE,V_CONTENT       - 文本专用标题字段、内容字段
    V_FEATURES              - 分类专用特征列
    O_ID                    - 输出ID
    O_TARGET                - 输出目标值

变量说明
    df前缀（函数形参）
    df_data                 - 数据集
    df_train                - 训练集
    df_feature              - 训练集数据规范化过后
    df_test                 - 测试集（考试使用的验证集）
    df_true                 - 真实值
    df_pred                 - 预测值

i前缀（从文件输入数据专用）
    i_data                  - （输入）数据集
    i_train                 - （输入）训练集
    i_feature               - （输入）训练集数据规范化过后
    i_test                  - （输入）测试集（考试使用的验证集）
    i_true                  - （输入）真实值
    i_pred                  - （输入）预测值

文件规范（文件后缀）
    _train.csv              - 训练文件集，exam.csv_train()
    _train_feature.csv      - 训练文件集（规范化之后），exam.csv_feature()
    _test.csv               - 测试集（考试使用的验证集），exam.csv_test()
    _target.csv             - 验证集真实值，exam.target()

代码文件（文件前缀）
    actor_                  - Actor：调度类，用于调度不同流程
                            - Answer：统计类，统计不同算法执行结果
    estimator_              - 自定义Estimator
    model_                  - 不同算法的基本模型
    pre_                    - 预处理专用，cat为分类，txt为文本
    toolkit_common          - 通用工具集
               _i_          - 内部工具集
               _c_          - 特定工具集

RunPhase，整体执行流程
                Splitting    ->  Pre   ->  Model  ->  Predict -->  Score
      Split        o
      Pre                         o
      Model                                  o
      Predict                                           o
      Score                                                          o
      Mix_SP       o              o
      Mix_MT                                 o          o
      Mix_MTE                                o          o            o   （保留）

Actor中有二阶函数用于绑定不同阶段的函数，但Mix_类型的混合阶段不支持绑定

特定函数（前缀）
    data_：公用处理数据前缀
    txt_：文本分析类
    cat_：分类问题类

_fn后缀为二阶函数，直接和一阶函数对应，二阶方便 Actor 绑定

内部变量
    x_train: 训练集X
    x_test: 验证集X
    y_train: 训练集Y
    y_test: 验证集Y

项目结构
    调度层：Actor - 主程序调度，Answer - 报表生成
    自定义组件（不对外）：estimator_前缀
    模型层：model_ 和 modelm_ 建模算法拆分
以上类都只导入：from examination.toolkit import *
"""

# Toolkit of Internal
from examination.toolkit import *

# Pre
from examination.phase_cat import *
from examination.phase_txt import *
from examination.phase_reg import *
from examination.phase_common import *

# Actor
from examination.actor_runner import *
from examination.actor_report import *
from examination.actor_function import *
from examination.actor_dq import *

# Modeling
from examination.model_xgboost import *
from examination.model_svc import *
from examination.model_mlp import *
from examination.model_lightbgm import *
from examination.model_logistic import *
from examination.model_dtc import *
from examination.model_rf import *
from examination.model_rfxgb import *
from examination.model_catboost import *

# Modeling Multi Values
from examination.modelv_xgboost import *
from examination.modelv_lightgbm import *
from examination.modelv_rf import *
from examination.modelv_rfxgb import *
from examination.modelv_catboost import *

# Modeling Multi Columns
from examination.modelm_lightgbm import *
from examination.modelm_xgboost import *

# Modeling Multi Label
from examination.modell_lightgbm import *

# Modeling Regression
from examination.modelr_xgboost import *
from examination.modelr_lightgbm import *
from examination.modelr_rf import *
from examination.modelr_catboost import *
