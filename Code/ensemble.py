import numpy as np
import pandas as pd
import sys
import glob
import pickle
import os
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from preprocess_global_weather_data import get_max_doy
from output_predictions import output_model
from visualize_output import plot_hist_residuals
from visualize_output import plot_clear_sky_index_residuals
from date_functions import get_doy
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


def aggregate_models(dirs_models, training_or_prediction, out_file):

    if os.path.isfile(settings.OUTDIR + out_file):
        with open(settings.OUTDIR + out_file) as f:
            all_models, y_train = pickle.load(f)
    else:
        # Grab the directories and stack the models horizontally.
        all_models = None
        for dirs_ in dirs_models:
            avg_model_, y_train = avg_models(dirs_, training_or_prediction)
            if all_models is None:
                all_models = avg_model_
            else:
                all_models = np.vstack((all_models, avg_model_))
        with open(settings.OUTDIR + out_file, 'w') as f:
            pickle.dump([all_models, y_train], f)
    return all_models, y_train


def avg_linear_regression_models(features, targets, features_sub,
                                 nsamp, train_frac):
    y_pred = None
    y_pred_sub = None
    for rands in range(nsamp):

        indices = np.arange(features.shape[0])
        # Step 2, do the linearly optimized fit
        linear_reg = LinearRegression()

        id_train, id_valid = train_test_split(indices, train_size=train_frac,
                                              random_state=rands)

        features_in = np.hstack((features[id_train, :],
                                 features[id_train, :]**2))
        linear_reg.fit(features_in, targets[id_train])

        if y_pred is None:
            features_in = np.hstack((features, features**2))
            y_pred = linear_reg.predict(features_in)
        else:
            features_in = np.hstack((features, features**2))
            y_pred = y_pred + linear_reg.predict(features_in)

        # Output the linear regression model
        if y_pred_sub is None:
            features_in = np.hstack((features_sub, features_sub**2))
            y_pred_sub = linear_reg.predict(features_in)
        else:
            features_in = np.hstack((features_sub, features_sub**2))
            y_pred_sub = y_pred_sub + linear_reg.predict(features_in)

    return y_pred / float(nsamp), y_pred_sub / float(nsamp)


def median_percent_diff(targets, predictions):
    percent_diff_mad = np.median((targets - predictions) / targets)
    return percent_diff_mad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    argsall = ['--indirtag', '--outdir', '--tag']

    for ar in argsall:
        parser.add_argument(ar)
    args = parser.parse_args()

    dirs = glob.glob(args.indirtag+'*/')
    settings.OUTDIR = os.getcwd() + '/' + args.outdir + '/'

    # Step 1: aggregate the training data
    all_models_train, y_train = aggregate_models(dirs, 'training',
                                                 'all_models_train.pickle')
    all_models_train = all_models_train.T

    # Step 2:
    all_models_submit, _ = aggregate_models(dirs, 'prediction',
                                            'all_models_submit.pickle')
    all_models_submit = all_models_submit.T

    with open(dirs[0] + args.indirtag + '_0' + '.pickle') as f:
        _, statnums, longs, lats, elevs, date, meth, debug = pickle.load(f)
    with open(dirs[0] + 'bad_data.pickle') as f:
        bad_data = pickle.load(f)
    statnums = np.delete(statnums, bad_data)
    date = np.delete(date, bad_data)
    with open(dirs[0] + settings.NORMCOEFFS_FILE) as f:
        x_coeff, y_coeff, w_ymax = pickle.load(f)

    statnums = statnums[w_ymax]
    date = date[w_ymax]
    doy = get_doy(date)

    with open(dirs[0] + args.indirtag + '_0' + '_test.pickle') as f:
        statnums_sub, longs, lats, elevs, date, meth, debug = pickle.load(f)

    y_preds = np.zeros(y_train.shape[0])
    y_preds_sub = np.zeros(all_models_submit.shape[0])

    # Now correct for any global bias in each station individually.
    for i in np.unique(statnums):
        w1 = np.where(statnums == i)[0]
        w2 = np.where(statnums_sub == i)[0]

        y_pred, y_pred_sub = avg_linear_regression_models(
            all_models_train[w1, :], y_train[w1],
            all_models_submit[w2, :] / y_coeff,
            100, 0.95)

        mad = (1.0 + median_percent_diff(y_train[w1], y_pred))

        y_preds[w1] = mad * y_pred
        y_preds_sub[w2] = mad * y_pred_sub * y_coeff

        plot_hist_residuals(y_preds[w1], y_train[w1], 'final_residuals'
                            + str(i) + '.png')

    plot_hist_residuals(y_train, y_preds, 'final_residuals.png')

    w1 = np.where(statnums < 100)[0]
    clear_sky_index_true = y_train / get_max_doy(doy, y_train)

    plot_clear_sky_index_residuals(y_train[w1], y_preds[w1],
                                   clear_sky_index_true[w1],
                                   'clear_sky_index_resids.png', 100)

    output_model(y_preds_sub, settings.OUTDIR
                 + 'avg_bystation_linear_reg_ensemble_submit.csv')
