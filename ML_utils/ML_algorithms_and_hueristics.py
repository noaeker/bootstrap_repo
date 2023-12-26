from side_code.config import *
from ML_utils.ML_config import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import matthews_corrcoef, log_loss, brier_score_loss, roc_auc_score, average_precision_score
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score
import lightgbm
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix


def regression_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return r2_score(y, y_pred)


def performance_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return matthews_corrcoef(y, y_pred)


def AUPRC_per_group(df):
    y_pred = list(df.y_pred)
    y = list(df.y)
    return average_precision_score(y, y_pred)


def score_func(y, y_pred, classification, groups_data):
    all_grouping = []
    for group in groups_data:
        df = pd.DataFrame({'y': y, 'y_pred': y_pred, 'grouping_col': groups_data[group]})
        if classification:
            df = df.groupby('grouping_col').apply(performance_per_group).reset_index(name='mcc')
        else:
            df = df.groupby('grouping_col').apply(regression_per_group).reset_index(name='R2')
        df["grouping_col_name"] = group
        all_grouping.append(df)
    return pd.concat(all_grouping)

class RfePipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_

def RFE(model, X, y, group_splitter, n_jobs, scoring, do_RFE):
    if do_RFE:
        min_features = 1
    else:
        min_features = X.shape[1]
    if model is None:
        model = RfePipeline([('sts', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=0))])
    RFE_res = RFECV(model, step=1, cv=group_splitter, n_jobs=n_jobs, min_features_to_select=min_features,
                 scoring=scoring)  # min_features_to_select= 30,X.shape[1] X.shape[1]
    selector =  RFE_res.fit(X, y.ravel())
    X_new = X[X.columns[selector.get_support(indices=True)]]
    logging.info(f"Number of features after feature selection: {X_new.shape[1]} out of {(X.shape[1])}")
    return selector, X_new



def ML_training(X_train, groups, y_train, n_jobs, path, classifier=False, model='lightgbm', calibrate=True, name="", large_grid = False, do_RFE = False, n_cv_folds = 3):
    logging.info(f"Building a {name} model and saving to {path}")
    if path and os.path.exists(path):
        logging.info(f"Using existing model in {path}")
        model = pickle.load(open(path, "rb"))
        return model
    else:
        group_splitter = list(GroupKFold(n_splits=n_cv_folds).split(X_train, y_train.ravel(), groups=groups))
        if classifier:
            if model=='lightgbm':
                model =lightgbm.LGBMClassifier(importance_type='gain',metric='binary_logloss')#importance_type='gain'
                param_grid = LIGHTGBM_CLASSIFICATION_PARAM_GRID
                if large_grid:
                    param_grid.update(GENERAL_PARAM_GRID)
            elif model=='NN':
                model = make_pipeline(StandardScaler(),MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes = (30, 5), random_state = 1))
                param_grid = {'mlpclassifier__hidden_layer_sizes':[(5,2),(25,5),(50,10)],'mlpclassifier__alpha': [0.0001,0.05]}
            elif model == 'rf':
                model = RandomForestClassifier()
                param_grid = {'max_depth': [3, 5, 10],'min_samples_split': [2, 5, 10]}
        else:
            model = lightgbm.LGBMRegressor()
            param_grid = REGRESSION_PARAM_GRID
            if large_grid:
                REGRESSION_PARAM_GRID.update(GENERAL_PARAM_GRID)
        if classifier:
            scoring = 'neg_log_loss'
        else:
            scoring = 'r2'
        selector, X_train = RFE(model, X_train, y_train, group_splitter, n_jobs, scoring, do_RFE = do_RFE)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   cv=group_splitter, n_jobs=n_jobs-4, pre_dispatch='1*n_jobs', verbose=2,
                                   scoring=scoring)
        grid_search.fit(X_train, y_train.ravel())
        best_model = grid_search.best_estimator_
    if classifier and calibrate:
        logging.info("Generating calibrated model for classification model")
        calibrated_model = CalibratedClassifierCV(base_estimator=best_model, cv=group_splitter, method='isotonic')
        calibrated_model.fit(X_train, y_train.ravel())
    else:
        calibrated_model = best_model
    model = {'best_model': best_model, 'calibrated_model': calibrated_model, 'selector': selector}
    if path:
        pickle.dump(model, open(path, "wb"))
    return model


def calibration_plot(model, test_data, y_test):
    calibrated_prediction = model['calibrated_model'].predict_proba((model['selector']).transform(test_data))[:, 1]
    uncalibrated_prediction = model['best_model'].predict_proba((model['selector']).transform(test_data))[:, 1]
    fop_uncalibrated, mpv_uncalibrated = calibration_curve(y_test, uncalibrated_prediction, n_bins=10, normalize=True)
    fop_calibrated, mpv_calibrated = calibration_curve(y_test, calibrated_prediction, n_bins=10)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--', color='black')
    # plot model reliabilities
    pyplot.plot(mpv_uncalibrated, fop_uncalibrated, marker='.', color = 'blue')
    pyplot.plot(mpv_calibrated, fop_calibrated, marker='.', color = 'green')
    pyplot.show()

def enrich_with_single_feature_metrics(var_impt, train_X, y_train, test_X, y_test):
        curr_mcc_scores = []
        curr_auc_scores = []
        for feature in var_impt.index:
            lg = lightgbm.LGBMClassifier(importance_type='gain').fit(train_X[[feature]], y_train)
            pred_prob= lg.predict_proba(test_X[[feature]])[:, 1]
            pred = lg.predict_proba(test_X[[feature]])[:, 1] >= 0.5
            curr_mcc_scores.append(matthews_corrcoef(y_test, pred))
            auc = roc_auc_score(y_test,pred_prob)
            curr_auc_scores.append(auc)

        var_impt[f"mcc"] = curr_mcc_scores
        var_impt[f"auc"] = curr_auc_scores


def model_evaluation_metrics(y_true, prob_predictions, estimate_auc = True):
    metrics = {}
    thresholds = [0.5,0.95]
    for threshold in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=prob_predictions>=threshold, labels=[True, False]).ravel()
        mcc = matthews_corrcoef(y_true=y_true, y_pred = prob_predictions>=threshold )
        t_metrics = {f'tn_{threshold}': tn, f'fp_{threshold}': fp,f'fn_{threshold}': fn,f'tp_{threshold}': tp, f'mcc_{threshold}': mcc}
        metrics.update(t_metrics)
    if estimate_auc:
        prob_metrics = {'AUC': roc_auc_score(y_true, prob_predictions),'brier_loss': brier_score_loss(y_true, prob_predictions),
                        'logloss': log_loss(y_true, prob_predictions), 'average_precision': average_precision_score(y_true, prob_predictions), }
        metrics.update(prob_metrics)
    return metrics


def model_groups_anlaysis(X,y_true, predictions):
    groups_data_test = X[
        ["feature_msa_n_seq", "feature_msa_n_loci", "feature_msa_pypythia_msa_difficulty"]]  #

    groups_dict_test = {'msa_difficulty_group': pd.qcut(groups_data_test["feature_msa_pypythia_msa_difficulty"], 4),
                        "n_seq_group": pd.qcut(groups_data_test["feature_msa_n_seq"], 4),
                        "feature_msa_n_loci": pd.qcut(groups_data_test["feature_msa_n_loci"], 4)}

    performance_per_group = score_func(y_true, predictions, classification=True, groups_data=groups_dict_test)
    return performance_per_group


def overall_model_performance_analysis(working_dir, model, data_dict, ML_model, name):
    if model and ML_model=='lightgbm':
        var_impt = variable_importance((data_dict["train"]["X"]).columns[model['selector'].get_support(indices=True)], model['best_model'])
        enrich_with_single_feature_metrics(var_impt, data_dict["train"]["X"], data_dict["train"]["y"], data_dict["test"]["X"], data_dict["test"]["y"])
        vi_path = os.path.join(working_dir, f'{name}_vi.tsv')
        var_impt.to_csv(vi_path, sep=CSV_SEP)
    all_metrics = pd.DataFrame()
    all_group_metrics = pd.DataFrame()
    prob_predictions_per_dataset = {}
    for dataset in data_dict:
        if model:
            prob_predictions =  model['calibrated_model'].predict_proba((model['selector']).transform(data_dict[dataset]["X"]))[:, 1]
        else:
            raw_bootstrap_values = data_dict[dataset]["X"].drop(['ignore'],axis=1).iloc[:,0]
            prob_predictions = raw_bootstrap_values
        true_label = data_dict[dataset]["y"]
        evaluation_metrics = model_evaluation_metrics(true_label, prob_predictions)
        evaluation_metrics["dataset"] = dataset
        evaluation_metrics["name"] = name
        evaluation_metrics["metric_type"] = "all_data"
        logging.info(f"Model evaluation metrics {evaluation_metrics}")
        all_metrics = all_metrics.append(evaluation_metrics, ignore_index= True)
        group_performance = model_groups_anlaysis(data_dict[dataset]["full_data"], data_dict[dataset]["y"], prob_predictions>0.5)
        group_performance["dataset"] = dataset
        group_performance["name"] = name
        all_group_metrics = pd.concat([all_group_metrics,group_performance])
        prob_predictions_per_dataset[dataset] = prob_predictions
    return all_metrics,all_group_metrics, prob_predictions_per_dataset



def train_test_validation_splits(full_data, test_pct, subsample_train=False,
                                 subsample_train_frac=-1):
    logging.info(f"Original number of trees in full data is {len(full_data.tree_id.unique())}")
    np.random.seed(SEED)
    logging.info("Partitioning MSAs according to number of sequences")
    msa_n_seq_group = pd.qcut(full_data["feature_msa_n_seq"], 3)
    full_sampling_data = pd.DataFrame({'msa_n_seq': msa_n_seq_group, 'tree': full_data['true_tree_path']}).drop_duplicates().sample(frac=1,random_state = SEED)
    full_sampling_data  = full_sampling_data.sort_values('tree') # Sort according to MSAs
    test_trees = full_sampling_data.groupby('msa_n_seq').sample(frac=test_pct, random_state= SEED)
    train_trees = full_sampling_data.loc[~full_sampling_data['tree'].isin(test_trees['tree'])]
    if subsample_train:
        logging.info(f"Subsampling training data to {subsample_train_frac}")
        train_trees = train_trees.groupby('msa_n_seq').sample(frac=subsample_train_frac, random_state=SEED)

    train_data = full_data[full_data['true_tree_path'].isin(train_trees['tree'])]
    logging.info(f"Number of MSAs in training data is {len(train_data.msa_path.unique())}")
    #logging.info(f"Number of overall positive samples in train: {len(train_data.loc[train_data.is_global_max == 1].index)}, Number of overall negative samples in test {len(train_data.loc[train_data.is_global_max == 0].index)}")
    test_data = full_data[full_data['true_tree_path'].isin(test_trees['tree'])]
    logging.info(f"Number of MSAs in test data is {len(test_data.msa_path.unique())}")
    #logging.info(f"Number of overall positive samples in test: {len(test_data.loc[test_data.is_global_max == 1].index)}, Number of overall negative samples in test {len(test_data.loc[test_data.is_global_max == 0].index)}")
    return train_data, test_data


def variable_importance(columns, model):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(columns, model.feature_importances_):
        feats[feature] = importance  # add the name/value pair

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace=True)
    return importances
