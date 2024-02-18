from exam_kit import *

ex.Answer() \
    .put("XGBoost", ex.ModRXGBoost) \
    .put("LightGBM", ex.ModRLightGBM) \
    .put("Catboost", ex.ModRCatboost) \
    .run_reg(mix_modeling)