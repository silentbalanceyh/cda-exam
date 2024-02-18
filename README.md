# CDA Level III 实战

## 1. 基础环境

* 环境地址：<https://www.anaconda.com/download>

> 推荐直接使用 Anaconda 环境而不是自己搭建相关环境，此环境已经包含了大量数据分析类所需的库相关信息，而基本分析IDE环境推荐直接使用 PyCharm 或 DataSpell（ JetBrains ）。

## 2. 依赖库

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

安装完成后，执行根目录的 version.py 脚本查看核心库的对应版本。



## 3. 考试库