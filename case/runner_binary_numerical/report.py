from exam_kit import *

ex.Answer() \
    .put("Logistic", ex.ModLogistic) \
    .put("DTC", ex.ModDtc) \
    .put("Catboost", ex.ModCatboost) \
    .put("LightGBM", ex.ModLightGBM) \
    .put("XGBoost", ex.ModXGBoost) \
    .put("Mlp", ex.ModMLP) \
    .put("RForest", ex.ModRForest) \
    .put("RForestXGB", ex.ModRForestXGB) \
    .run_cat(mix_modeling)