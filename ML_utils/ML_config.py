GENERAL_PARAM_GRID = {
    'n_estimators': [100,300],
    'max_depth': [-1,3,6,12],
    'num_leaves': [25, 50, 100, 200],
    'learning_rate':[0.001,0.01,0.1],
    #'reg_alpha': [0, 10, 25, 50, 100],
    #'reg_lambda': [0, 10, 25, 50, 100],
    #'min_split_gain': [0.3, 0.4],
    'subsample': [0.6, 0.8, 1],
    'subsample_freq': [20],
    #'boosting_type': ['gbdt','rf','dart']
}


LIGHTGBM_CLASSIFICATION_PARAM_GRID = {
    'metric': ['auc'],
    'objective': ['binary']
}
#CLASSIFICATION_PARAM_GRID.update(GENERAL_PARAM_GRID)


REGRESSION_PARAM_GRID = {
    'metric': ['rmse'],
    'objective': ['regression'],

}
#REGRESSION_PARAM_GRID.update(GENERAL_PARAM_GRID)


