# CDA Level III 实战

## 1. 基础环境

### 1.1. Anaconda 下载

* 环境地址：<https://www.anaconda.com/download>

> 推荐直接使用 Anaconda 环境而不是自己搭建相关环境，此环境已经包含了大量数据分析类所需的库相关信息，而基本分析IDE环境推荐直接使用 PyCharm 或 DataSpell（ JetBrains ）。

### 1.2. PPT 下载

## 2. 依赖库

### 2.1. 安装脚本

```bash
# （一）基本运行库
pip install pyecharts

# （二）分词专用
# Jieba分词，必须带 -c，否则无法查找此库
conda install -c conda-forge jieba
# Jieba快速版本（考试可考虑，只能使用 pip 的方式安装）
pip install jieba_fast

# （三）算法模型库
# XGBoost
conda install xgboost
# CatBoost
conda install catboost
# LightBGM
```

### 2.2. 版本检查

安装完成后，执行根目录的 `version.sh` 脚本查看核心库的对应版本。

```bash
./version.sh
# 输出信息如
xgboost -> 1.7.3
catboost -> 1.2
lightgbm -> 4.1.0
sklearn -> 1.2.2
imblearn -> 0.11.0
gensim -> 4.3.0
pandas -> 2.1.4
numpy -> 1.26.3
```

## 3. 考试库

### 3.1. 库用法

考试库有两个用途：

* 直接重用：如果您对整个 `examination` 部分已经十分熟悉了，那么可以直接重用（调参需要自行操作）。
* 了解流程：您也可以自己写属于自己的完整流程的代码来执行。

### 3.2. 基本代码介绍

|命名前缀| 含义                               |
|:---|:---------------------------------|
|`actor_`| 主要执行器，用于执行代码、生成报表、打印结果、生成质量报告专用。 |
|`estimator_`| 特征处理中的值编码器，填充、均衡、编码专用（预处理部分）。    |
|`phase_` | 标准化处理流程，分类标准化、回归标准化、文本分词标准化。     |
|`model_`| 单属性、单值模型。                        |
|`modelv_`| Values -- 单属性、多值模型。              |
|`modelm_`| Multi -- 多属性、单值模型。               |
|`modelr_`| Regression -- 回归模型。              |
|`modell_`| Label -- 多标签模型。                  |
|`toolkit`| 底层工具包，类定义。                       |
|`toolkit_i_`| （toolkit内部使用）功能包：类定义、文件读写、日志记录。  |
|`toolkit_o_`| （多被 phase 外部调用）功能包：分词、数据集划分、分词处理、矩阵运算。|

### 3.3. 入口配置

考试库中最核心的一个类为 `Actor`（ `actor_runner.py` 中定义 ）