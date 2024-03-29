# CDA Level III 实战

## 1. 基础环境

### 1.1. Anaconda 下载

* 环境地址：<https://www.anaconda.com/download>

> 推荐直接使用 Anaconda 环境而不是自己搭建相关环境，此环境已经包含了大量数据分析类所需的库相关信息，而基本分析IDE环境推荐直接使用
> PyCharm 或 DataSpell（ JetBrains ）。

### 1.2. PPT 下载

* [01.基本准备](ppt/01.基本准备.pptx)
* [02.二分类-非数值输入](ppt/02.二分类-非数值输入.pptx)
* [03.二分类-数值输入](ppt/03.二分类-数值输入.pptx)
* [04.二分类-文本输入](ppt/04.二分类-文本输入.pptx)
* [05.回归](ppt/05.回归.pptx)

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

| 命名前缀         | 含义                                     |
|:-------------|:---------------------------------------|
| `actor_`     | 主要执行器，用于执行代码、生成报表、打印结果、生成质量报告专用。       |
| `estimator_` | 特征处理中的值编码器，填充、均衡、编码专用（预处理部分）。          |
| `phase_`     | 标准化处理流程，分类标准化、回归标准化、文本分词标准化。           |
| `model_`     | 单属性、单值模型。                              |
| `modelv_`    | Values -- 单属性、多值模型。                    |
| `modelm_`    | Multi -- 多属性、单值模型。                     |
| `modelr_`    | Regression -- 回归模型。                    |
| `modell_`    | Label -- 多标签模型。                        |
| `toolkit`    | 底层工具包，类定义。                             |
| `toolkit_i_` | （toolkit内部使用）功能包：类定义、文件读写、日志记录。        |
| `toolkit_o_` | （多被 phase 外部调用）功能包：分词、数据集划分、分词处理、矩阵运算。 |

## 4. 使用

```python
import examination as ex
```

如果环境配置正确，那么您就可以直接使用上述代码导入 `examination` 的专用库（自定义）。

### 4.1. 案例结构

目录和文件说明

| 目录/文件            | 含义                                 |
|:-----------------|:-----------------------------------|
| data             | 数据文件目录。                            |
| model            | 建模专用目录。                            |
| runtime          | 运行专用目录。                            |
| `__init.py__`    | Python 语言规范下的 package 专用目录。        |
| `constant.py`    | 可配置的常量定义。                          |
| `exam_kit.py`    | 考试工具，`runner` 会调用定义的函数，方便读者了解内部流程。 |
| `exam_runner.py` | （主脚本）执行脚本。                         |
| `report.py`      | 多算法比较目录，在部分场景中可对比不同算法针对数据集的结果。     |

案例说明

| 案例                          | 含义                   |
|:----------------------------|:---------------------|
| `runner_binary_categorical` | 二分类案例，常用Excel标准输入格式。 |
| `runner_binary_numerical`   | 二分类案例，数值输入格式。        |

### 4.2. 执行流程

执行步骤如下：

1. 打开 `exam_runner.py` 脚本。
2. 将五个步骤对应的内容注释掉：

    ```python
    # 步骤1 - Splitting
    # run_splitting()
    
    # 步骤2 - Feature
    # run_feature()
    
    # 步骤3 - Modeling
    # run_model()
    
    # 步骤4 - Predict
    # run_predict()
    
    # 步骤5 - Score
    # run_score()
    ```
3. 最终考试的时候，将原始数据集直接执行 `run_predict()` 的预测即可。