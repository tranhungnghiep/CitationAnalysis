"""
Using simple features for predicting paper ranking in venue.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import time
import sys
import argparse

import pandas as pd
import numpy as np
np.random.seed(7)
import scipy.stats
import random
random.seed(7)
import pickle

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.pipeline
import sklearn.preprocessing, sklearn.model_selection, sklearn.metrics, sklearn.feature_selection
import sklearn.linear_model, sklearn.svm, sklearn.kernel_ridge, sklearn.ensemble, sklearn.neural_network

import multiprocessing
import joblib

from prepare_data_highimpact import read_simple_x, read_highimpact_y


def main(args):

    print('START.')
    start_time_main = time.time()  # in second.

    predict_high_impact(args)

    print('FINISH.')
    stop_time_main = time.time()
    print('Time (s): ' + str(stop_time_main-start_time_main))

    return 0


def predict_high_impact(args):
    """
    OUTLINE:
    - Load data.
    - Preprocess:
        - match x, y and clean split train/test.
        - compute sample weight.
    - Feature analysis:
        - compute feature variance.
        - correlation between each feature and output: plot (raw value) and compute pearson r.
        - statistical feature analysis:
            - F-test: only work well for linear relationship.
            - Mutual information: work for nonlinear relationship too.

    * START PIPELINE.
    - Transform
        - scale to standard.
        - onehotencoder Month, isConf.
    - Estimator: train model. Try some models:
        - Linear model:
            - Linear regression. -> just use LR directly.       O(d^2*m)
            - Elastic net (mix Ridge, Lasso). -> through SGDRegressor.
            - Linear SVR. -> through SGDRegressor.      O(d*m)
        - Nonlinear model: (this problem has small feature set, kernel tricks may run fast enough.)
            (- Kernel SVR: polinomial/rbf.)     O(n^3) or O(d*m^2)?
            (- Kernel ridge: no need, same as svr.)
            - Random forest regression.     O(T*d*mlogm)
            - MLP regression.       O(d*i*o*h^k*m)
            - Deep MLP regression (in keras).
    - Grid search CV. (need scoring function here.)

    - Model debug:
        - Check overfitting/underfitting:
            plot learning curve to check overall
            and validation curve to check each hyperparameter if it causes over/underfitting.
        - Check stable:
            Check results of CV: small std means stable. (reCV or use grid.cv_results_)
    - Feature selection and analysis:
        - manually drop each feature and do train/test.
    - Model evaluation: ontrain and ontest.
        - y is citation count.
        - y is rank in venue and predict 2 step.
        - y is rank in venue and predict directly.
    """

    global debug

    # LOAD DATA.
    paper_align_filename = os.path.join(args.root_path, args.paper_align_filename)
    venue_paper_rank_filename = os.path.join(args.root_path, args.venue_paper_rank_filename)

    paper_simple_filename = os.path.join(args.root_path, args.paper_simple_filename)
    author_simple_filename = os.path.join(args.root_path, args.author_simple_filename)
    venue_simple_filename = os.path.join(args.root_path, args.venue_simple_filename)

    features_used = 'Simple_features'
    inter_dir = '_'.join([features_used, str(args.start_test_year), str(args.end_test_year), str(args.period),
                          'trim' + ''.join(map(str, args.trim_combine)), 'fill' + args.fillna, 'pooling' + ''.join(args.pooling),
                          'minpaper' + str(args.min_paper_count)])
    save_dir = os.path.join(args.root_path, args.save_dir, inter_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    temp_dir = os.path.join(args.root_path, args.temp_dir, inter_dir)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    # Read.
    base_time = time.time()
    df_xtrain, df_xtest = read_simple_x(paper_align_filename, paper_simple_filename=paper_simple_filename, author_simple_filename=author_simple_filename, venue_simple_filename=venue_simple_filename,
                                        pooling=args.pooling, start_test_year=args.start_test_year, end_test_year=args.end_test_year, period=args.period, trim_combine=args.trim_combine, fillna=args.fillna)
    print('Read X: {}'.format(str(time.time() - base_time))); base_time = time.time()
    df_y = read_highimpact_y(venue_paper_rank_filename, min_paper_count=args.min_paper_count)  # e.g., get the y data to train by df_y.loc[:, ['Paper_ID', 'Size', 'Rank']]
    print('Read Y: {}'.format(str(time.time() - base_time))); base_time = time.time()
    # Filter out x without y.
    df_xtrain = df_xtrain.loc[df_xtrain.loc[:, 'Paper_ID'].isin(df_y.loc[:, 'Paper_ID'].values), :].sort_values('Paper_ID')
    df_xtest = df_xtest.loc[df_xtest.loc[:, 'Paper_ID'].isin(df_y.loc[:, 'Paper_ID'].values), :].sort_values('Paper_ID')
    # Split train test y.
    df_ytrain = df_y.loc[df_y.loc[:, 'Paper_ID'].isin(df_xtrain.loc[:, 'Paper_ID'].values), :].sort_values('Paper_ID')
    df_ytest = df_y.loc[df_y.loc[:, 'Paper_ID'].isin(df_xtest.loc[:, 'Paper_ID'].values), :].sort_values('Paper_ID')
    print('Train x shape: ' + str(df_xtrain.shape))  # CHECKPOINT.
    print('Train y shape: ' + str(df_ytrain.shape))  # CHECKPOINT.
    print('Test x shape: ' + str(df_xtest.shape))  # CHECKPOINT.
    print('Test y shape: ' + str(df_ytest.shape))  # CHECKPOINT.
    # More checkpoint.
    assert((df_xtrain.loc[:, 'Paper_ID'].values == df_ytrain.loc[:, 'Paper_ID'].values).all())
    assert((df_xtest.loc[:, 'Paper_ID'].values == df_ytest.loc[:, 'Paper_ID'].values).all())

    # Compute sample_weight for citcount. Prefer large citcount, log scale.
    # Use info in train, not in test data.
    # Note that sample weights in sklearn influence the loss of each sample, but not directly, and depending on each method. For SVM, it changes C = C * sample_weight.
    min_value = df_ytrain.loc[:, 'Size'].min()
    max_value = df_ytrain.loc[:, 'Size'].max()
    log_min_value = np.log(1 + min_value)
    log_max_value = np.log(1 + max_value)
    range_value = log_max_value - log_min_value
    df_ytrain.loc[:, 'sw_citcount'] = [(np.log(1 + x) - log_min_value) / range_value for x in df_ytrain.loc[:, 'Size'].values] if range_value != 0 else 1
    df_ytest.loc[:, 'sw_citcount'] = [(np.log(1 + x) - log_min_value) / range_value for x in df_ytest.loc[:, 'Size'].values] if range_value != 0 else 1
    # Compute sample_weight for rank. Prefer small rank, linear scale.
    min_value = df_ytrain.loc[:, 'Rank'].min()
    max_value = df_ytrain.loc[:, 'Rank'].max()
    range_value = max_value - min_value
    df_ytrain.loc[:, 'sw_rank'] = [(1 + max_value - x) / range_value for x in df_ytrain.loc[:, 'Rank'].values] if range_value != 0 else 1
    df_ytest.loc[:, 'sw_rank'] = [(1 + max_value - x) / range_value for x in df_ytest.loc[:, 'Rank'].values] if range_value != 0 else 1

    # Save df.
    df_xtrain.to_csv(os.path.join(temp_dir, 'TRAINX.txt'), sep=' ', header=True, index=False)
    df_ytrain.to_csv(os.path.join(temp_dir, 'TRAINY.txt'), sep=' ', header=True, index=False)
    df_xtest.to_csv(os.path.join(temp_dir, 'TESTX.txt'), sep=' ', header=True, index=False)
    df_ytest.to_csv(os.path.join(temp_dir, 'TESTY.txt'), sep=' ', header=True, index=False)

    # Feature analysis.
    # Variance, other statistics. Non-numeric columns will be ignored.
    df_xtrain.describe().to_csv(os.path.join(save_dir, 'TRAINX' + '_' + 'STATS.csv'), sep=' ', header=True, index=True)
    df_ytrain.describe().to_csv(os.path.join(save_dir, 'TRAINY' + '_' + 'STATS.csv'), sep=' ', header=True, index=True)
    # Pearson. For each feature in xtrain and each output in ytrain.
    pearson = df_xtrain.iloc[:, 1:].corrwith(df_ytrain.loc[:, 'Size']).to_frame(name='Citcount').T \
                       .append(df_xtrain.iloc[:, 1:].corrwith(df_ytrain.loc[:, 'Rank']).to_frame(name='Rank').T) \
                       .reset_index()
    pearson.to_csv(os.path.join(save_dir, 'TRAIN' + '_' + 'PEARSON.csv'), sep=' ', header=True, index=False)


    # PIPELINE.
    # Preprocess.
    pipe_preprocess = [
        # Remove ID
        ('remove_id', PandasColumnRemover(key=['Paper_ID', 'Year'])),

        # Remove each feature, default: remove nothing
        ('leave_feature', PandasColumnRemover(key=[])),

        # Transform and Union
        ('union', sklearn.pipeline.FeatureUnion(
            transformer_list=[
                # Scale numeric features
                ('scale', sklearn.pipeline.Pipeline(steps=[
                    ('remove', PandasColumnRemover(key=['Month', 'isConf'])),
                    ('scale', sklearn.preprocessing.StandardScaler())
                ])),
                # Dummy Month
                ('dummy_month', sklearn.pipeline.Pipeline(steps=[
                    ('select', PandasColumnSelector(key=['Month'])),
                    ('dummy', sklearn.preprocessing.OneHotEncoder(n_values=13, handle_unknown='ignore'))
                ])),
                # Keep as is: Dummy isConf
                ('keep', sklearn.pipeline.Pipeline(steps=[
                    ('select', PandasColumnSelector(key=['isConf']))
                ]))
            ],
            n_jobs=-1
        ))
    ]

    # Estimator.
    pipes_estimator = {
        # Linear.
        # Basic baseline
        'lr': [('estimator', sklearn.linear_model.LinearRegression())],

        # Ridge, Lasso, ElasticNet
        'sgdr': [('estimator', sklearn.linear_model.SGDRegressor(
            loss='squared_loss',
            penalty='elasticnet', alpha=1e-2, l1_ratio=0.15,
            learning_rate='invscaling', eta0=0.01, power_t=0.25,
            epsilon=0.1,
            fit_intercept=True, shuffle=True, random_state=7, n_iter=args.iter, verbose=1
        ))],

        # Linear SVR
        'lsvr': [('estimator', sklearn.svm.LinearSVR(
            loss='epsilon_insensitive',
            epsilon=0.0,
            C=1e2,
            fit_intercept=True, intercept_scaling=1.0,
            random_state=7, tol=1e-3, max_iter=args.iter, verbose=1
        ))],

        # None linear.
        # SVR
        'svr': [('estimator', sklearn.svm.SVR(
            epsilon=0.1,
            C=1e2,
            kernel='rbf', gamma='auto',
            degree=3, coef0=1.0,
            shrinking=True, cache_size=2000, tol=1e-3, max_iter=args.iter, verbose=1
        ))],

        # Random forest
        'rf': [('estimator', sklearn.ensemble.RandomForestRegressor(
            n_estimators=100,
            criterion='mse',
            max_features=1.0, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_split=1e-3,
            bootstrap=True, oob_score=False, n_jobs=-1, random_state=7, verbose=1
        ))],

        # MLP, default loss function is MSE
        'mlp': [('estimator', sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh',
            alpha=1e-2,
            solver='adam', learning_rate_init=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
            learning_rate='constant', power_t=0.5, momentum=0.9, nesterovs_momentum=True,
            shuffle=True, batch_size='auto',
            early_stopping=False, validation_fraction=0.1,
            random_state=7, tol=1e-3, max_iter=args.iter, verbose=1
        ))]
    }

    # Pipeline and hyperparams.
    estimator_uses = ['lr', 'sgdr', 'lsvr', 'svr', 'rf', 'mlp']
    pipeline = sklearn.pipeline.Pipeline(pipe_preprocess + pipes_estimator[estimator_uses[args.method_used]])

    # Params for gridsearch cv here
    if estimator_uses[args.method_used] == 'sgdr':
        param_grid = {
            'estimator__alpha': [1, 0.1, 0.01, 0.001, 0.0001],
            'estimator__l1_ratio': [0, 0.15, 0.5, 0.85, 1]
        }
    elif estimator_uses[args.method_used] == 'lsvr':
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100, 1000, 10000],
            'estimator__epsilon': [0, 0.1, 1, 10, 100, 1000]
        }
    elif estimator_uses[args.method_used] == 'svr':
        param_grid = {
            'estimator__C': [0.1, 1, 10, 100],
            'estimator__epsilon': [0.1, 1, 10, 100]
        }
    elif estimator_uses[args.method_used] == 'rf':
        param_grid = {
            'estimator__n_estimators': [10, 50, 100, 500]
        }
    elif estimator_uses[args.method_used] == 'mlp':
        param_grid = {
            'estimator__hidden_layer_sizes': [(100,), (200,), (50, 50)],
            'estimator__activation': ['tanh', 'logistic', 'relu']
        }
    else:
        param_grid = None

    # Params for fit method, note: not for estimator constructor but fit method, that is sample_weight.
    output_uses = ['Size', 'Rank']
    sw_uses = ['sw_citcount', 'sw_rank']
    if 'sw' in args.config:
        fit_param = {
            'estimator__sample_weight': df_ytrain.loc[:, sw_uses[args.out_used]]
        }
    else:
        fit_param = None

    # Grid search.
    grid = sklearn.model_selection.GridSearchCV(pipeline, param_grid=param_grid, fit_params=fit_param, scoring=args.scoring, cv=args.cv, refit=True, n_jobs=-1, verbose=0)
    grid.fit(df_xtrain, df_ytrain.loc[:, output_uses[args.out_used]])

    cv_result = pd.DataFrame(grid.cv_results_)
    cv_result.to_csv(os.path.join(save_dir, 'CV_RESULT' + '_' + '{}_{}.csv'.format(estimator_uses[args.method_used], output_uses[args.out_used])), sep=' ', header=True, index=False)
    print(cv_result)

    return 0


class PandasColumnSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    PandasColumnSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = PandasColumnSelector(key='a')
    >> data['a'] == ds.transform(data)

    PandasColumnSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Author: Matt Terry <matt.terry@gmail.com>
    License: BSD 3 clause

    Parameters
    ----------
    key : hashable, list/iterable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        valid_key = [k for k in self.key if k in x.columns]
        return x.loc[:, valid_key]


class PandasColumnRemover(BaseEstimator, TransformerMixin):
    """For data grouped by feature, remove subset of data at a provided key.

    Parameters
    ----------
    key : hashable, list/iterable, required
        The key will be removed.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        valid_key = [k for k in self.key if k in x.columns]
        return x.drop(valid_key, axis=1)


def log_scale(y):
    """
    :param y: all values in y will be transformed, so y must not contain id, only values.
    """

    return np.log(y.astype(float) + 1)


def inverse_log_scale(y):
    """
    :param y: all values in y will be transformed, so y must not contain id, only values.
    """

    return np.exp(y.astype(float)) - 1


def evaluate(regressor, testx, testy, transformer_x, transformer_y, save_dir, model_name, note='None'):
    """
    Input: fitted model.
    Predict. Remember to round value and save predicted values.
    Evaluate. Save evaluation result.
    Note:
        input: numpy array.
        :return: result on every year step.
    """

    model_name = str(model_name) + '_' + str(note)
    print('EVALUATE: ' + model_name)
    start_time = time.time()  # in second.

    # Save ground truth.
    print('Min, max value in testy: ' + str(testy[:, 1:].min()) + ', ' + str(testy[:, 1:].max()))
    np.savetxt(os.path.join(save_dir, 'groundtruth_' + model_name + '.csv'), np.concatenate((testy[:, [0]], testy[:, 1:].astype(float)), axis=1), fmt='%s %20.0f %20.0f %20.0f %20.0f %20.0f', delimiter='\t')  # Use astype float to preserve original magnitude, int could be overflow. Save format %20.0f to show all float number and no decimal, for visibility.

    # predict
    predicty = regressor.predict(testx[:, 1:])
    # Inverse transform.
    if transformer_y is not None:
        predicty = transformer_y.inverse_transform(predicty)
        predicty = np.nan_to_num(predicty)  # fix numeric unstability.
    # Post-process.
    predicty = np.round(predicty)  # round citcount.
    # Save original predict.
    print('Min, max value in predicty orig: ' + str(predicty[:, 1:].min()) + ', ' + str(predicty[:, 1:].max()))
    np.savetxt(os.path.join(save_dir, 'predict_' + model_name + '_orig' + '.csv'), np.concatenate((testx[:, [0]], predicty.astype(float)), axis=1), fmt='%s %20.0f %20.0f %20.0f %20.0f %20.0f', delimiter='\t')  # DEBUG.
    # Further post-process.
    predicty[predicty < 0] = 0
    # Save final predict.
    print('Min, max value in predicty: ' + str(predicty[:, 1:].min()) + ', ' + str(predicty[:, 1:].max()))
    np.savetxt(os.path.join(save_dir, 'predict_' + model_name + '.csv'), np.concatenate((testx[:, [0]], predicty.astype(float)), axis=1), fmt='%s %20.0f %20.0f %20.0f %20.0f %20.0f', delimiter='\t')
    # Add paper id.
    predicty = np.concatenate((testx[:, [0]], predicty), axis=1)  # Slicing return 2D array, indexing return 1D, 2D array is needed for concatenate().

    # evaluate: only evaluate each year step, then take average by .mean().
    r2 = sklearn.metrics.r2_score(testy[:, 1:], predicty[:, 1:], multioutput='raw_values')
    mse = sklearn.metrics.mean_squared_error(testy[:, 1:], predicty[:, 1:], multioutput='raw_values')
    num_year_step = testy.shape[1] - 1
    rho, p = np.zeros(num_year_step), np.zeros(num_year_step)
    for i in range(0, num_year_step):
        rho[i], p[i] = scipy.stats.pearsonr(testy[:, i+1], predicty[:, i+1])
    stop_time = time.time()

    # save evalutation result
    evaluation_result_file = os.path.join(save_dir, 'evaluation_result.csv')
    with open(evaluation_result_file, "a+") as f:  # create if not exist
        if os.path.getsize(evaluation_result_file) == 0:  # write header if empty
            header = 'Model\t' + 'Evaluation_time\t' + 'Year_step\t' + 'R^2\t' + 'MSE\t' + 'Pearson_rho\t' + 'p-value\n'
            f.write(header)
        for i in range(0, num_year_step):
            evaluation_result = model_name + '\t' + str(stop_time - start_time) + '\t' + str(i+1) + '\t' + str(r2[i]) + '\t' + str(mse[i]) + '\t' + str(rho[i]) + '\t' + str(p[i]) + '\n'
            f.write(evaluation_result)
        avg_evaluation_result = model_name + '\t' + str(stop_time - start_time) + '\t' + 'Average' + '\t' + str(r2.mean()) + '\t' + str(mse.mean()) + '\t' + str(rho.mean()) + '\t' + str(p.mean()) + '\n'
        f.write(avg_evaluation_result)

    print('R^2: ' + str(r2.mean()))
    print('MSE: ' + str(mse.mean()))
    print('Pearson rho: ' + str(rho.mean()) + ', p-value: ' + str(p.mean()))

    print('EVALUATE: DONE.')
    print('Time (s): ' + str(stop_time-start_time))

    return 0


def parse_args():
    """Parses the arguments.
    """

    global debug  # claim to use global var.

    parser = argparse.ArgumentParser(description="Step 1: rank paper in venue using simple features.")

    # nargs='+': take 1 or more arguments, return a list. '+' == 1 or more. '*' == 0 or more. '?' == 0 or 1.
    parser.add_argument('--method-used', type=int, default=0,
                        help='Method. Default 0: lr.')

    parser.add_argument('--out-used', type=int, default=0,
                        help='Output data. Default 0: citcount.')

    parser.add_argument('--config', nargs='*', default=[],
                        help='More config, e.g., sw for sample weight. Default: [].')

    parser.add_argument('--root-path', default=None,
                        help="Root folder path. Default None.")

    parser.add_argument('--save-dir', default='Save',
                        help="Save folder. Default 'Save'.")

    parser.add_argument('--temp-dir', default='temp',
                        help='Temp folder. Default "temp".')

    parser.add_argument('--load-data', nargs='*', default=[],
                        help='Load processed data in temp dir or not. List of string: x, y. Default []: not load, read and save.')

    parser.add_argument('--paper-align-filename', default='PAPER_ALIGN_1970_2005.txt')
    parser.add_argument('--venue-paper-rank-filename', default='VENUE_PAPER_RANK_PERIOD5.txt')

    parser.add_argument('--paper-simple-filename', default='PAPER_SIMPLE.txt')
    parser.add_argument('--author-simple-filename', default='AUTHOR_SIMPLE.txt')
    parser.add_argument('--venue-simple-filename', default='VENUE_SIMPLE.txt')

    parser.add_argument('--start-test-year', type=int, default=1996,
                        help='The start test year. Default: 1996.')
    parser.add_argument('--end-test-year', type=int, default=2000,
                        help='The end test year. Default: 2000.')
    parser.add_argument('--period', type=int, default=5,
                        help='Period after publication. Default: 5.')

    parser.add_argument('--transform-x', default='raw',
                        help="How to transform x: {'raw', 'scaling', 'normalizing', 'mix'}. Default 'raw'.")

    parser.add_argument('--transform-y', default='raw',
                        help="How to transform y: {'raw', 'scaling', 'log'}. Default 'raw'.")

    parser.add_argument('--pooling', nargs='+', default=['avg', 'max'],
                        help="How to pool author embeddings: ['avg', 'sum', 'max']. Default ['avg', 'max'].")

    parser.add_argument('--trim-combine', nargs='*', default=[1, 1, 0],
                        help="List of 1/0 values. Default [1, 1, 0] means [trim by paper, trim by author, no trim by venue].")

    parser.add_argument('--fillna', default='0',
                        help='How to fillna. {"avg": Fill by average of each column; or a value}. Default 0.')

    parser.add_argument('--min-paper-count', type=int, default=10,
                        help='Min paper in each venue. Default: 10.')

    parser.add_argument('--parallel', type=int, default=multiprocessing.cpu_count() - 2,
                        help='Number of parallel jobs. Default cpu count - 2.')

    parser.add_argument('--cv', type=int, default=2,
                        help='Number of k folds cross validation. Default: 2: same complexity as not cv, but still have generalization ability.')

    parser.add_argument('--iter', type=int, default=1000,
                        help='Number of iteration when optimizing methods. Default: 1000.')

    parser.add_argument('--scoring', default='r2',
                        help='How to score grid search CV. Default: r2 score for regression: same as estimator. \n\
                        Note that score is not loss when training. Scoring is after training. Loss for regression is usually MSE (e.g., in SGDRegressor). Score is usually r2, pearson...')

    parser.add_argument('--debug-server', dest='debug_server', action='store_true',
                        help='Turn on debug mode on server. Default: off.')
    parser.set_defaults(debug_server=False)

    args = parser.parse_args()

    # Post-process args value.
    if args.root_path is None:
        root_path_local = "./data/HighImpact/MAG"
        root_path_server = "~/Data/HighImpact/MAG"
        if os.path.isdir(root_path_local):
            args.root_path = root_path_local
            debug = True
        else:
            args.root_path = root_path_server
    if not os.path.isdir(os.path.join(args.root_path, args.save_dir)):
        os.makedirs(os.path.join(args.root_path, args.save_dir))
    if not os.path.isdir(os.path.join(args.root_path, args.temp_dir)):
        os.makedirs(os.path.join(args.root_path, args.temp_dir))

    # Finally return.
    return args


if __name__ == '__main__':
    debug = False
    main(parse_args())
