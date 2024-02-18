from enum import Enum

from abc import abstractmethod, ABCMeta


class ModeOutlier(Enum):
    MaxMin = "M"  # 过大/过小值
    Cap = "C"  # 平均值法
    Quartile = "Q"  # 四分位法
    Trust = "T" # 95%,5% 的盖帽法


class ModeAnalyzer(Enum):
    Numeric = "numeric"  # 数值型
    Categorical = "categorical"  # 类别型


class EncoderType(Enum):
    Binary = "Binary"  # 二分类
    Values = "Values"  # 多分类：一列多值（多列双值可还原）
    Columns = "Columns"  # 多标签：多列多值


class CaseType(Enum):
    Numeric = "Numeric"  # 二分类（侧重数值）
    Binary = "Binary"  # 二分类（侧重类别）
    Multi = "Multi"  # 多分类（二维表）
    Textual = "Textual"  # 二分类（文本）
    TextualMulti = "TextualMulti",    # 多分类（文本）
    LabelMulti = "MultiLabel",  # 多标签（侧重类别）
    Regression = "Regression"   # 回归问题


class RunPhase(Enum):
    Split = "Splitting"  # 拆分
    Pre = "Pre"  # 预处理
    Model = "Modeling"  # 建模
    Predict = "Predict"  # 预测
    Score = "Score"  # 评分
    # 混合步骤
    Mix_SP = "Splitting,Pre"
    Mix_MT = "Modeling,Predict"
    Mix_MTE = "Modeling,Predict,Score"


class Mod:
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute(self, executor):
        pass
