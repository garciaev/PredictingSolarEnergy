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
from fit_data import fit_data
import settings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    argsall = ['--outdir', '--modelnum', '--numclosegrid', '--debug',
               '--method', '--numrandstate', '--tag']
    for ar in argsall:
        parser.add_argument(ar)
    args = parser.parse_args()

    out_dir = settings.ROOT_DIR + str(args.outdir) + '/'

    # Load in the GEFS files to be used
    train_gefs_files = np.sort(glob.glob(settings.DATA_DIR_TRAIN + '*nc'))
    test_gefs_files = np.sort(glob.glob(settings.DATA_DIR_TEST + '*nc'))

    # Load in station info
    station_info = pd.read_csv(settings.STATION_INFO_CSV)

    if os.path.isdir(out_dir) is False:
        print 'Making output directory ' + out_dir
        os.mkdir(out_dir)
    print 'Saving to directory ' + str(args.outdir)

    # Specify which model to use
    model_num = np.int(args.modelnum)
    print 'Using model ' + str(model_num)

    # Specify the number of grid points to use for each station
    nclose = np.int(args.numclosegrid)
    print 'Using ' + str(nclose) + ' closest grid points'

    # Specify whether debugging or not
    debug = bool(np.int(args.debug))
    print 'Debugging mode = ' + str(debug)

    # Current data preprocessing methods. Average global weather models,
    # or let boosted trees method find the best fit?
    meth = str(args.method)  # avg, wavg
    print 'Using method = ' + meth

    # Number of different random states to run
    nrand = np.int(args.numrandstate)
    print 'Number of random states = ' + str(nrand)

    # Name of files that will be saved
    out_tag_name = out_dir + args.tag


    #---------------
    # DATA PIPELINE.
    #----------------
    # 1. Get the .NC files which contain the weather forecasts from
    # 11 different global numerical weather prediction models. Average
    # the data spatially.
    print "Make or grab the training data, averaging spatially..."
    trainX, trainY, testX = assemble_data(out_tag_name, meth, debug, nclose,
                                          station_info, model_num,
                                          train_gefs_files, test_gefs_files)
    print 'Training sizes:'
    print trainX.shape, trainY.shape


    # 2. Normalize the weather variables, use
    # both train and testing X data for the
    # X normalization to fully encompass the range of X values.
    trainX, trainY, testX, xcoeff, ycoeff = normalize_data(trainX, trainY,
                                                           testX, out_dir)

    # 3. Fit the data
    fit_data(trainX, trainY, nrand, out_dir, ycoeff, testX)
    print 'Finished.'
