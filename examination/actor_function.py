from examination.phase_common import *
from examination.phase_txt import *
from examination.phase_cat import *
from examination.phase_reg import *
from examination.actor_runner import *
def report_reg_tpl(keys=None):
    if keys is None:
        keys = []
    return {
        "Algorithm": keys,
        "Sec": [],
        "Explained": [],
        "Absolute": [],
        "Squared": [],
        "Median": [],
        "R2": []
    }


def report_reg_add(tpl, row):
    tpl['Sec'].append(row['duration'])
    tpl['Explained'].append(row['explained'])
    tpl['Absolute'].append(row['absolute'])
    tpl['Squared'].append(row['squared'])
    tpl['Median'].append(row['median'])
    tpl['R2'].append(row['r2'])

def report_cat_tpl(keys=None):
    if keys is None:
        keys = []
    return {
        "Algorithm": keys,
        "Sec": [],
        "Acc": [],
        "F1_Mi": [],
        "F1_Ma": [],
        "P_Mi": [],
        "P_Ma": [],
        "R_Mi": [],
        "R_Ma": []
    }


def report_cat_add(tpl, row):
    tpl['Sec'].append(row['duration'])
    tpl['Acc'].append(row['accuracy'])
    tpl['F1_Mi'].append(row['f1_micro'])
    tpl['F1_Ma'].append(row['f1_macro'])
    tpl['P_Mi'].append(row['precision_micro'])
    tpl['P_Ma'].append(row['precision_macro'])
    tpl['R_Mi'].append(row['recall_micro'])
    tpl['R_Ma'].append(row['recall_macro'])


def report_txt(modeler, f_id, f_target, o_id, o_target, o_filename=None, f_classes=None):
    i_feature = csv_feature()
    i_test = csv_test()
    runner = Actor(f_id, f_target, f_classes=f_classes).fn_run_before(data_modeling_fn).fn_run(modeler)
    runner.fn_predict(lambda df_test, filename: txt_predict_fn(
        df_test=df_test,
        f_model=filename,
        o_id=o_id,
        o_target=o_target,
        o_filename=o_filename
    ))
    df_predict, duration = runner.execute(i_feature, RunPhase.Mix_MT, i_test)
    # 预测
    df_true = csv_target()
    params = cat_score(df_true, df_predict, f_target, o_target)
    params['duration'] = duration
    return params

def report_reg(modeler, f_id, f_target, o_id, o_target, f_features, o_filename=None):
    i_feature = csv_feature()
    i_test = csv_test()
    runner = Actor(f_id, f_target).fn_run_before(data_modeling_fn).fn_run(modeler)
    runner.fn_predict(lambda df_test, filename: reg_predict_fn(
        df_test=df_test,
        f_model=filename,
        f_categorical=f_features,
        o_id=o_id,
        o_target=o_target,
        o_filename=o_filename
    ))
    df_predict, duration = runner.execute(i_feature, RunPhase.Mix_MT, i_test)
    # 预测
    df_true = csv_target()
    params = reg_score(df_true, df_predict, f_target, o_target)
    params['duration'] = duration
    return params

def report_cat(modeler, f_id, f_target, o_id, o_target, f_features, o_filename=None, f_outlier=None):
    i_feature = csv_feature()
    i_test = csv_test()
    runner = Actor(f_id, f_target).fn_run_before(data_modeling_fn).fn_run(modeler)
    runner.fn_predict(lambda df_test, filename: cat_predict_fn(
        df_test=df_test,
        f_model=filename,
        f_categorical=f_features,
        o_id=o_id,
        o_target=o_target,
        o_filename=o_filename,
        f_outlier=f_outlier
    ))
    df_predict, duration = runner.execute(i_feature, RunPhase.Mix_MT, i_test)
    # 预测
    df_true = csv_target()
    params = cat_score(df_true, df_predict, f_target, o_target)
    params['duration'] = duration
    return params
