# -----------------------------------------------------------------------------------------------------
#
# 「导入区」
#
# -----------------------------------------------------------------------------------------------------
"""
Exam Tool（考试专用工具包）
导入自定义工具

「注」：
df_test 表示验证集
df_train 表示训练集
"""
import examination as ex

# -----------------------------------------------------------------------------------------------------
#
# 「数据输入区」
#
# -----------------------------------------------------------------------------------------------------
# -- 属性部分
# V_ID                                                      - 主键
# V_TARGET                                                  - 目标属性（单个）
# V_TARGETS                                                 - 目标属性（多个）
V_ID = "ID"
V_TARGET = "Target"
V_TARGETS = ["none", "disgust", "like", "happiness", "sadness", "surprise", "anger", "fear"]

# F_TITLE                                                   - 「文本」文本中的标题属性
# F_CONTENT                                                 - 「文本」文本中的内容属性
# F_FEATURES                                                - 「二维表」特征属性集
F_TITLE = None
F_CONTENT = "message"
F_FEATURES = []

# O_ID                                                      - 输出——主键
# O_TARGET                                                  - 输出——目标属性
O_ID = "ID"
O_TARGET = "Target"

# -- 文件部分
# IN_PRE                                                    - 处理前文件（执行预处理后生成 IN_SOURCE，有可能无此步骤）
# IN_SOURCE                                                 - 正式输入文件
# OUT_MODEL                                                 - 模型输出
# OUT_RESULT                                                - 结果输出
IN_PRE = "training.xlsx"
IN_SOURCE = "training-data.csv"

# -----------------------------------------------------------------------------------------------------
#
# 「算法选择区」
#
# -----------------------------------------------------------------------------------------------------
# 算法选择
CASE = ex.CaseType.TextualMulti
MODELER = ex.ModVLightGBM
CLASSES = len(V_TARGETS)
OUT_MODEL = MODELER.__name__ + ".model"
OUT_RESULT = MODELER.__name__ + ".csv"
