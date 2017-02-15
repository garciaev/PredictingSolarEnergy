import sys
import glob
import os
import random
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import netCDF4 as nc
from preprocess_global_weather_data import get_gefs_features
from preprocess_global_weather_data import mae
from preprocess_global_weather_data import pd_chunk_csv
from preprocess_global_weather_data import create_feature_map
from grid_search_xgboost import grid_search_xgboost
from assemble_data import assemble_data
from normalize_data import normalize_data
import settings


def fit_data(trainX, trainY, nrand, out_dir, ycoeff, testX):
    """
    Fit the weather prediction data using either XGboost (decision trees)
    Neural networks, or both.
    :param trainX: weather prediction data
    :param trainY: actual solar energy measured at different mesonet stations
    :param nrand: number of times to run
    :param usenn: use Neural networks (Keras)
    :param out_dir: output directory.
    :return: nothin!
    """
    parameters = {"eta": [0.1],
                  "alpha": [1.0],
                  "lambda": [1.0],
                  "gamma": [1.0],
                  "colsample_bytree": [0.8],
                  "colsample_bylevel": [1.0],
                  "max_delta_step": [0.0],
                  "min_child_weight": [2.0],
                  "scale_pos_weight": [1.0],
                  "subsample": [0.8],
                  "max_depth": [25]}
    early_stop = settings.EARLY_STOP
    n_steps = settings.N_STEPS
    objective = settings.OBJECTIVE
    booster = settings.BOOSTER
    eval_metric = settings.EVAL_METRIC

    n_feats = trainX.shape[0]
    indices = np.arange(n_feats)
    # Split between train and evaluation data
    rands = [random.randint(0, 100000) for i in range(nrand)]
    # Loop over multiple random seeds
    for jj, rnd in enumerate(rands):
        rnd = 63082
        rnd = 59035
        id_train, id_valid = train_test_split(indices,
                                              train_size=settings.TRAIN_FRAC,
                                              random_state=rnd)
        print 'Running XGBoost grid...'
        print trainX[id_train].shape
        grid_search_xgboost(trainX[id_train], trainY[id_train],
                            trainX[id_valid], trainY[id_valid],
                            parameters, n_steps, early_stop,
                            objective, booster, eval_metric,
                            rnd, out_dir, ycoeff, testX)

        model_tag = 'random_seed' + str(rnd)
        # Save the data for this run of the model.
        with open(out_dir + model_tag + '.pickle', 'w') as f:
            pickle.dump([nclose, model_num, rnd, indices, id_train,
                         id_valid], f)