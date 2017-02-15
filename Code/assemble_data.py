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
import settings


def assemble_data(out_tag_name, meth, debug, nclose, station_info, model_num,
                  train_gefs_files, test_gefs_files):
    """
    Take .nc files with weather prediction 12, 15, 18, 21, and 24 hours
    ahead weather predictions and make the the features for predicting
    day-ahead solar energy output from 98 weather stations in Oklahoma.
    :param out_tag_name:
    :param meth: Method of combining the day-ahead weather predictions
    :param debug: Whether to run a debugging session
    :param nclose: Number of grid points to use that are close
    :param station_info: The information from each pyranometer station
    :param model_num: Which global weather model to use.
    :return:
    """
    if os.path.isfile(out_tag_name+'.pickle'):
        print 'LOADING TRAINING DATA...'
        with open(out_tag_name + '.pickle', 'r') as f:
            trainY, statnums, longs, lats, elevs, date, meth, \
            debug = pickle.load(f)
            trainX = pd_chunk_csv(out_tag_name + '.csv', chunksize=10000)
    else:
        print 'GETTING DATA FROM .NC FILES...'
        trainY = pd.read_csv(settings.PREDICT_CSV)
        date = np.tile(trainY.values[:, 0], settings.NUM_STATIONS)
        # date is read in, get rid of it!
        trainY = np.float64(trainY.values[:, 1:])
        # Stack the stations, make sure ORDER IS PROPER.
        trainY = trainY.reshape(trainY.shape[0] * settings.NUM_STATIONS,
                                order='F')
        trainX, feats = get_gefs_features(model_num, nclose, train_gefs_files,
                                          station_info, method=meth,
                                          debug=debug)
        statnums = trainX[:, 0]
        elevs = trainX[:, 1]
        longs = trainX[:, 2]
        lats = trainX[:, 3]

        # Use lat, long, weather features.
        trainX = trainX[:, 2:]
        feats = feats[2:]

        # Output features and create feature map.
        create_feature_map(settings.FEATURE_NAMES_FILE, feats)

        with open(out_tag_name + '.pickle', 'w') as f:
            pickle.dump([trainY, statnums, longs, lats, elevs, date, meth,
                         debug], f)
        pd.DataFrame(trainX, columns=feats).to_csv(out_tag_name + '.csv',
                                                       index=False)
    # Do the same for the testing data.
    if os.path.isfile(out_tag_name + '_test.pickle'):
        print 'No need to load test data.'
        testX = pd_chunk_csv(out_tag_name + '_test.csv', chunksize=10000)
    else:
        print 'Getting data from files...'
        testX, feats = get_gefs_features(model_num, nclose,
                                             test_gefs_files,
                                             station_info, method=meth,
                                             debug=debug)
        statnums_t = testX[:, 0]
        elevs_t = testX[:, 1]
        longs_t = testX[:, 2]
        lats_t = testX[:, 3]

        testX = testX[:, 2:]
        date_t = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV,
                             usecols=(0,)).values.flatten()
        feats = feats[2:]

        with open(out_tag_name + '_test.pickle', 'w') as f:
            pickle.dump([statnums_t, longs_t, lats_t, elevs_t, date_t, meth,
                         debug], f)
        pd.DataFrame(testX, columns=feats).to_csv(out_tag_name + '_test.csv',
                                                  index=False)
    return trainX, trainY, testX
