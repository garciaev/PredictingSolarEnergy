import numpy as np
import pandas as pd
import sys
import glob
import pickle
import os
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from preprocess_global_weather_data import mae, mae_percent
from preprocess_global_weather_data import mad_percent
from preprocess_global_weather_data import create_feature_map
from output_predictions import output_model
from fit_data import fit_data
import settings


def avg_models(dir_, training_or_prediction):
    out_data = None
    if training_or_prediction == 'prediction':
        all_files = glob.glob(dir_ + '/*submit.csv')
        for csv_files in all_files:
            if out_data is None:
                out_data = pd.read_csv(csv_files)
            else:
                out_data = out_data + pd.read_csv(csv_files)
        out_data = out_data / np.float(len(all_files))
        out_data.to_csv(dir_ + '/submit_averaged.csv')
        out_data = out_data.values[:, 1:].reshape(1796 * 98, order='F')
        return out_data, None

    if training_or_prediction == 'training':
        all_files1 = glob.glob(dir_ + '/parms*pickle')
        all_files2 = glob.glob(dir_ + '/*ids_random_seed*pickle')
        if len(all_files1) != len(all_files2):
            print 'Something is wrong?'
            sys.exit()

        for pickle_files1, pickle_files2 in zip(all_files1, all_files2):

            with open(pickle_files1, 'r') as f:
                params_cur, objective, random_seed, early_stop, nsteps, \
                evals_result, npts, model_tag, y_train, y_train_pred, y_valid, \
                y_valid_pred = pickle.load(f)
            with open(pickle_files2, 'r') as f:
                tag_name, model_tag, rnd, indices, id_train, id_valid, feats, \
                jj = pickle.load(f)

            y_true = np.zeros(y_train.shape[0] + y_valid.shape[0])
            y_preds = np.zeros(y_train.shape[0] + y_valid.shape[0])

            if y_preds.shape[0] != id_train.shape[0] + id_valid.shape[0]:
                print 'Something is wrong!'
                sys.exit()

            y_preds[id_train], y_preds[id_valid] = y_train_pred, y_valid_pred
            y_true[id_train], y_true[id_valid] = y_train, y_valid

            if out_data is None:
                out_data = y_preds
            else:
                out_data = out_data +  y_preds
        out_data = out_data / np.float(len(all_files1))
        return out_data, y_true


def aggregate_models(dirs_models, training_or_prediction):
    # Grab the directories and stack the models horizontally.
    all_models = None
    for dirs_ in dirs_models:
        avg_model_, y_train = avg_models(dirs_, training_or_prediction)
        if all_models is None:
            all_models = avg_model_
        else:
            all_models = np.vstack((all_models, avg_model_))
    if training_or_prediction == 'training':
        return all_models, y_train
    if training_or_prediction == 'prediction':
        return all_models, y_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    argsall = ['--outdirtag']

    for ar in argsall:
        parser.add_argument(ar)
    args = parser.parse_args()

    dirs = glob.glob(args.outdirtag+'*/')

    # Step 1, aggregate the training data
    all_models_train, y_train = aggregate_models(dirs, 'training')
    all_models_train = all_models_train.T
    indices = np.arange(y_train.shape[0])

    # Step 2, do the linearly optimized fit
    linear_reg = LinearRegression()
    id_train, id_valid = train_test_split(indices, train_size=0.9,
                                          random_state=5)

    linear_reg.fit(all_models_train[id_train, :], y_train[id_train])

    y_preds = linear_reg.predict(all_models_train[id_valid, :])
    y_preds_mean = all_models_train[id_valid, :].mean(axis=1)

    print 'MAE, MAE Percent, MAD Percent with linear regression'
    print mae(y_preds, y_train[id_valid])
    print mae_percent(y_preds, y_train[id_valid])
    print mad_percent(y_preds, y_train[id_valid])

    print 'MAE, MAE Percent, MAD Percent with straightforward average'
    print mae(y_preds_mean, y_train[id_valid])
    print mae_percent(y_preds_mean, y_train[id_valid])
    print mad_percent(y_preds_mean, y_train[id_valid])

    # Step 3:
    all_models_submit, _ = aggregate_models(dirs, 'prediction')
    all_models_submit = all_models_submit.T

    # Output the linear regression model
    y_preds = linear_reg.predict(all_models_submit)
    output_model(y_preds, 'linear_reg_ensemble_submit.csv')

    # Output the straightforward average model
    y_preds_mean = all_models_submit.mean(axis=1)
    output_model(y_preds_mean, 'mean_ensemble_submit.csv')

    # Perform a grid search using XGBoost.
    settings.N_STEPS = 400
    settings.EARLY_STOP = 50
    settings.ETA = [0.05],
    settings.ALPHA = [0.0],
    settings.LAMBDA = [0.0],
    settings.GAMMA = [0.0],
    settings.COLSAMPLE_BYTREE = [1.0],
    settings.COLSAMPLE_BYLEVEL = [1.0],
    settings.MAX_DELTA_STEP = [0.0],
    settings.MIN_CHILD_WEIGHT = [1.0],
    settings.SCALE_POS_WEIGHT = [1.0],
    settings.SUBSAMPLE = [1.0],
    settings.MAX_DEPTH = [2, 3, 4, 5, 7, 8, 9]
    settings.OUTDIR = os.getcwd()+'/ensemble1/'
    settings.FEATURE_NAMES_FILE = 'level2features.csv'
    settings.TRAIN_FRAC = 0.9

    num_rand = 1
    feats = ['Model' + str(num) for num in range(all_models_submit.shape[1])]
    create_feature_map(settings.OUTDIR + settings.FEATURE_NAMES_FILE, feats)

    # Get the y coefficient
    with open(dirs[0] + settings.NORMCOEFFS_FILE, 'r') as f:
        x_coeff, y_coeff = pickle.load(f)

    print 'Fitting level 2 model ensemble...'
    fit_data(all_models_train, y_train, num_rand, y_coeff,
             all_models_submit / y_coeff,
             'ensemble_take1', False)
