"""
Multimodal features using node2vec embeddings for predicting citation count.
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
import sklearn.preprocessing
import sklearn.linear_model, sklearn.kernel_ridge, sklearn.svm, sklearn.multioutput
import sklearn.metrics

import joblib

from prepare_data_citcount import read_x, read_y


def main(args):
    """
    Outline:
    - read paper list, read embedding, read citation sequence.
        -> get trainx, trainy, testx, testy.
        -> preprocess feature.
    - regression.
        -> fit, save model.
        -> predict, save value.
        -> evaluate, save result.
    """

    global debug

    print('START.')
    start_time_main = time.time()  # in second.


    # Data.
    paper_align_filename = os.path.join(args.root_path, args.paper_align_filename)
    citation_count_filename = os.path.join(args.root_path, args.citation_count_filename)

    emb_dir = 'Embeddings'
    if 'mag7l' in args.config:
        emb_choice = 'MAG7L'
    else:
        emb_choice = 'MAG7'  # Default.
    inter_input_dir = os.path.join(emb_dir, emb_choice)
    if not os.path.isdir(os.path.join(args.root_path, inter_input_dir)):
        os.makedirs(os.path.join(args.root_path, inter_input_dir))

    inter_save_dir = emb_choice + '_' + str(args.test_year) + '_' + ''.join(map(str, args.input))  # Default: trim combine, avg pooling: not noted.
    inter_temp_dir = emb_choice + '_' + str(args.test_year) + '_' + ''.join(map(str, args.input))  # Default: trim combine, avg pooling: not noted.
    if not args.trim_combine:
        inter_save_dir = inter_save_dir + '_' + 'notrim' + '_' + 'fill' + args.fillna
        inter_temp_dir = inter_temp_dir + '_' + 'notrim' + '_' + 'fill' + args.fillna
    if args.pooling == 'sum' or args.pooling == 'max':
        inter_save_dir = inter_save_dir + '_' + args.pooling
        inter_temp_dir = inter_temp_dir + '_' + args.pooling
    save_dir = os.path.join(args.root_path, args.save_dir, inter_save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    temp_dir = os.path.join(args.root_path, args.temp_dir, inter_temp_dir)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)

    if 'undirect123' in args.config:
        direction_choice = 'undirect123'
    else:
        direction_choice = 'direct123'  # Default.
    if debug:
        inter_input_dir = ''
        direction_choice = ''
        args.input = [1, 2, 3]
    paper_cit_filename = os.path.join(args.root_path, inter_input_dir, direction_choice, args.paper_cit_filename)
    author_cit_filename = os.path.join(args.root_path, inter_input_dir, direction_choice, args.author_cit_filename)
    venue_cit_filename = os.path.join(args.root_path, inter_input_dir, direction_choice, args.venue_cit_filename)
    paper_sa_filename = os.path.join(args.root_path, inter_input_dir, args.paper_sa_filename)
    author_sp_filename = os.path.join(args.root_path, inter_input_dir, args.author_sp_filename)
    author_sv_filename = os.path.join(args.root_path, inter_input_dir, args.author_sv_filename)
    venue_sa_filename = os.path.join(args.root_path, inter_input_dir, args.venue_sa_filename)

    if 1 not in args.input:
        paper_cit_filename = None
    if 2 not in args.input:
        author_cit_filename = None
    if 3 not in args.input:
        venue_cit_filename = None
    if 4 not in args.input:
        paper_sa_filename = None
    if 5 not in args.input:
        author_sp_filename = None
    if 6 not in args.input:
        author_sv_filename = None
    if 7 not in args.input:
        venue_sa_filename = None

    if 'x' not in args.load_data:
        # Read x and save.
        trainx, testx = read_x(paper_align_filename=paper_align_filename,
                               paper_cit_filename=paper_cit_filename, author_cit_filename=author_cit_filename, venue_cit_filename=venue_cit_filename,
                               paper_sa_filename=paper_sa_filename, author_sp_filename=author_sp_filename,
                               author_sv_filename=author_sv_filename, venue_sa_filename=venue_sa_filename,
                               pooling=args.pooling, year=args.test_year, trim_combine=args.trim_combine, fillna=args.fillna)
        print('SAVE X.')
        start_time = time.time()  # in second.
        np.save(os.path.join(temp_dir, 'trainx.npy'), trainx)
        np.save(os.path.join(temp_dir, 'testx.npy'), testx)
        print('SAVE X: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time-start_time))
    else:
        print('LOAD X.')
        start_time = time.time()  # in second.
        trainx = np.load(os.path.join(temp_dir, 'trainx.npy'))
        testx = np.load(os.path.join(temp_dir, 'testx.npy'))
        print('LOAD X: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time - start_time))
    if 'y' not in args.load_data:
        # Read y and save.
        trainy = read_y(x=trainx, num_year_step=args.num_year_step, citation_count_filename=citation_count_filename)
        testy = read_y(x=testx, num_year_step=args.num_year_step, citation_count_filename=citation_count_filename)
        print('SAVE Y.')
        start_time = time.time()  # in second.
        np.save(os.path.join(temp_dir, 'trainy.npy'), trainy)
        np.save(os.path.join(temp_dir, 'testy.npy'), testy)
        print('SAVE Y: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time - start_time))
    else:
        print('LOAD Y.')
        start_time = time.time()  # in second.
        trainy = np.load(os.path.join(temp_dir, 'trainy.npy'))
        testy = np.load(os.path.join(temp_dir, 'testy.npy'))
        print('LOAD Y: DONE.')
        stop_time = time.time()
        print('Time (s): ' + str(stop_time-start_time))
    print('Train x shape: ' + str(trainx.shape))  # CHECKPOINT.
    print('Train y shape: ' + str(trainy.shape))  # CHECKPOINT.
    print('Test x shape: ' + str(testx.shape))  # CHECKPOINT.
    print('Test y shape: ' + str(testy.shape))  # CHECKPOINT.


    # Transform data.
    # For x, no need to inverse transform.
    transformer_x = None
    if args.transform_x == 'scaling':
        transformer_x = sklearn.preprocessing.StandardScaler()
        transformer_x.fit(trainx[:, 1:])
        trainx[:, 1:] = transformer_x.transform(trainx[:, 1:])
        testx[:, 1:] = transformer_x.transform(testx[:, 1:])
    elif args.transform_x == 'normalizing':
        transformer_x = sklearn.preprocessing.Normalizer()
        transformer_x.fit(trainx[:, 1:])
        trainx[:, 1:] = transformer_x.transform(trainx[:, 1:])
        testx[:, 1:] = transformer_x.transform(testx[:, 1:])
    elif args.transform_x == 'mix':
        pass  # Later, when using topic distribution.
    # For y, need to inverse transform in evaluate().
    transformer_y = None
    if args.transform_y == 'scaling':
        transformer_y = sklearn.preprocessing.StandardScaler()
        transformer_y.fit(trainy[:, 1:])
        trainy[:, 1:] = transformer_y.transform(trainy[:, 1:])
    elif args.transform_y == 'log':
        transformer_y = sklearn.preprocessing.FunctionTransformer(func=log_scale, inverse_func=inverse_log_scale, validate=False)
        trainy[:, 1:] = transformer_y.transform(trainy[:, 1:])


    # Model.
    note = args.transform_x + '_' + args.transform_y
    # Linear.
    if 1 in args.model:
        model_name = 'linear_multivariate'
        # There is option normalize=True. But better should try normalize beforehand for all methods.
        regressor = sklearn.linear_model.LinearRegression(n_jobs=-1)  # default fit_intercept=True, normalize=False, n_jobs=1.
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 2 in args.model:
        model_name = 'ridge_multivariate'
        regressor = sklearn.linear_model.Ridge()  # default regularization weight alpha=1, fit_intercept=True, normalize=False, max_iter=None, tol=0.001, solver='auto'.
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 3 in args.model:
        model_name = 'parallel_linear'
        estimator = sklearn.linear_model.LinearRegression(n_jobs=-1)
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 4 in args.model:
        model_name = 'parallel_ridge'
        estimator = sklearn.linear_model.Ridge()
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    # Non linear.
    if 5 in args.model:
        model_name = 'kernelridge_linear_multivariate'
        regressor = sklearn.kernel_ridge.KernelRidge(kernel='linear')  # kernel ridge is faster than svr on medium size dataset. multivariate result is not different to parallel, but maybe faster.
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 6 in args.model:
        model_name = 'kernelridge_rbf_multivariate'
        regressor = sklearn.kernel_ridge.KernelRidge(kernel='rbf')
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 7 in args.model:
        model_name = 'parallel_kernelridge_linear'
        estimator = sklearn.kernel_ridge.KernelRidge(kernel='linear')
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 8 in args.model:
        model_name = 'parallel_kernelridge_rbf'
        estimator = sklearn.kernel_ridge.KernelRidge(kernel='rbf')
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 9 in args.model:
        model_name = 'parallel_svr_linear'
        estimator = sklearn.svm.SVR(kernel='linear')  # default C=1, epsilon=0.1
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
    if 10 in args.model:
        model_name = 'parallel_svr_rbf'
        estimator = sklearn.svm.SVR(kernel='rbf')  # default C=1, epsilon=0.1
        regressor = sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=-1)
        regressor = train(regressor=regressor, trainx=trainx, trainy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)
        evaluate(regressor=regressor, testx=testx, testy=testy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note=note)


    print('FINISH.')
    stop_time_main = time.time()
    print('Time (s): ' + str(stop_time_main-start_time_main))

    return 0


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


def train(regressor, trainx, trainy, transformer_x, transformer_y, save_dir, model_name, note='None'):
    """
    Model, by sklearn.
    First try is:
     - linear regression (and l2 regularization ridge...):
        -> has built-in support for multi output: 1 model is fitted for many correlated dependent outputs,
        this technique is called "Multivariate linear regression" (https://en.wikipedia.org/wiki/General_linear_model, not to be confused with generalized linear model because this still uses least square loss)
        -> should try 5 independent regressors too.
     - svr: don't have multivariate output like the linear regression model
        -> 5 regression models for 5 years, parallel using MultiOutputRegressor.
        - linear kernel
        - rbf kernel

    Then, fit. Remember to save model.
    Note: input: numpy array.
    """

    model_name = str(model_name) + '_' + str(note)
    print('TRAIN: ' + model_name)
    start_time = time.time()  # in second.

    # fit
    regressor.fit(trainx[:, 1:], trainy[:, 1:])
    stop_time = time.time()

    # save model
    # pickle.dump(regressor, os.path.join(save_dir, 'model_' + model_name + '_' + str(stop_time-start_time) + '.pickle'))  # pickle is general purpose.
    joblib.dump(regressor, os.path.join(save_dir, 'model_' + model_name + '_' + str(stop_time-start_time) + '.joblib'))  # joblib is faster than pickle, especially for objects containing large numpy array.

    print('TRAIN: DONE.')
    print('Time (s): ' + str(stop_time-start_time))

    if transformer_y is None:
        evaluate(regressor=regressor, testx=trainx, testy=trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note='ontrain')
    else:  # inverse transform, avoid mutatively changing trainy.
        inverse_transform_trainy = transformer_y.inverse_transform(trainy[:, 1:])
        inverse_transform_trainy = np.round(inverse_transform_trainy.astype(float))  # Need to round, because after inverse transform there would be some small differences. Need to astype float because cannot round dtype float object.
        inverse_transform_trainy = np.concatenate((trainy[:, [0]], inverse_transform_trainy), axis=1)
        evaluate(regressor=regressor, testx=trainx, testy=inverse_transform_trainy, transformer_x=transformer_x, transformer_y=transformer_y, save_dir=save_dir, model_name=model_name, note='ontrain')

    return regressor


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

    parser = argparse.ArgumentParser(description="Level1.1: Multimodal.")

    # nargs='+': take 1 or more arguments, return a list. '+' == 1 or more. '*' == 0 or more. '?' == 0 or 1.
    parser.add_argument('--model', nargs='+', type=int, default=[0],
                        help='Model: {integer list, 0: all linear models, -1: all models, ...}. Default 0.')

    parser.add_argument('--input', nargs='+', type=int, default=[0],
                        help='Input file: {integer list for embeddings from network 1 to 7, 0: all 7 embs}. Default 0.')

    parser.add_argument('--config', nargs='*', default=[],
                        help='More config. E.g., mag7l, undirect123. Default: [].')

    parser.add_argument('--root-path', default=None,
                        help="Root folder path. Default None.")

    parser.add_argument('--save-dir', default='Save',
                        help="Save folder. Default 'Save'.")

    parser.add_argument('--temp-dir', default='temp',
                        help='Temp folder. Default "temp".')

    parser.add_argument('--load-data', nargs='*', default=[],
                        help='Load processed data in temp dir or not. List of string: x, y. Default []: not load, read and save.')

    parser.add_argument('--paper-align-filename', default='PAPER_ALIGN_1970_2005.txt')
    parser.add_argument('--citation-count-filename', default='CITATION_COUNT_1970_2005.txt')

    parser.add_argument('--paper-cit-filename', default='PAPER_CITATION_EMB_1996.txt')
    parser.add_argument('--author-cit-filename', default='AUTHOR_CITATION_EMB_1996_2.txt')
    parser.add_argument('--venue-cit-filename', default='VENUE_CITATION_EMB_1996_2.txt')
    parser.add_argument('--paper-sa-filename', default='PAPER_SHARE_AUTHOR_EMB_1996_2.txt')
    parser.add_argument('--author-sp-filename', default='AUTHOR_SHARE_PAPER_EMB_1996_2.txt')
    parser.add_argument('--author-sv-filename', default='AUTHOR_SHARE_VENUE_EMB_1996_5.txt')
    parser.add_argument('--venue-sa-filename', default='VENUE_SHARE_AUTHOR_EMB_1996_2.txt')

    parser.add_argument('--test-year', type=int, default=1996,
                        help='The year to test. Could only test each year. Default: 1996.')
    parser.add_argument('--num-year-step', type=int, default=5,
                        help='The number of year steps. Default: 5.')

    parser.add_argument('--transform-x', default='raw',
                        help="How to transform x: {'raw', 'scaling', 'normalizing', 'mix'}. Default 'raw'.")

    parser.add_argument('--transform-y', default='raw',
                        help="How to transform y: {'raw', 'scaling', 'log'}. Default 'raw'.")

    parser.add_argument('--pooling', default='avg',
                        help="How to pool author embeddings: {'avg', 'sum', 'max'}. Default 'avg'.")

    parser.add_argument('--trim-combine', dest='trim_combine', action='store_true',
                        help='[True] means trim papers when a combined embeddings lacking that paper id. False means keep all papers in paper_align, fillna in embedding by 0 or mean(). Default: trim.')
    parser.add_argument('--no-trim-combine', dest='trim_combine', action='store_false')
    parser.set_defaults(trim_combine=True)

    parser.add_argument('--fillna', default='avg',
                        help='How to fillna. {"avg": Fill by average of each column; or a value}. Default avg.')

    parser.add_argument('--debug-server', dest='debug_server', action='store_true',
                        help='Turn on debug mode on server. Default: off.')
    parser.set_defaults(debug_server=False)

    args = parser.parse_args()

    # Post-process args value.
    if 0 in args.model:
        # run all linear models.
        args.model = range(1, 5)
    elif -1 in args.model:
        # run all.
        args.model = range(1, 11)

    if 0 in args.input:
        # use all embeddings.
        args.input = range(1, 8)

    # Post-process args value.
    if args.root_path is None:
        root_path_local = "./data/CitationCount/MAG"
        root_path_server = "~/Data/CitationCount/MAG"
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
