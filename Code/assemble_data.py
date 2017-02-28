import os
import numpy as np
import pandas as pd
import pickle
from preprocess_global_weather_data import get_gefs_features
from preprocess_global_weather_data import pd_chunk_csv
from preprocess_global_weather_data import create_feature_map
from preprocess_global_weather_data import remove_bad_data
from visualize_output import plot_features_vs_targets
from visualize_output import plot_features_vs_targets_six
import settings


def assemble_data(out_tag_name, meth, debug, nclose, station_info, model_num,
                  train_gefs_files, test_gefs_files):
    """
    Take .nc files with weather prediction 12, 15, 18, 21, and 24 hours
    ahead weather predictions and make the the features for predicting
    day-ahead solar energy output from 98 weather stations in Oklahoma.
    The steps are:
    1. Spatially average the weather data each weather station
    2. Remove data points where pyranometer failed to get an accurate
    measurement of the actual solar energy that day.

    :param out_tag_name: the name to tag the output files
    :param meth: method of spatial averaging
    :param debug: whether to debug the code or not
    :param nclose: number of grid points to use
    :param station_info: a pandas data frame of information on each station
    from station_info.csv
    :param model_num: which global weather forecast model to use
    :param train_gefs_files: the input training .nc files
    :param test_gefs_files: the input testing .nc files.
    :return:
    """
    if os.path.isfile(settings.OUTDIR + out_tag_name + '.pickle'):
        print 'Loading up the previously assembled weather training data...'
        with open(settings.OUTDIR + out_tag_name + '.pickle', 'r') as f:
            trainY, statnums, longs, lats, elevs, date, meth, debug = \
                pickle.load(f)
        trainX = pd_chunk_csv(settings.OUTDIR + out_tag_name + '.csv',
                              chunksize=10000)
    else:
        print 'Assembling the weather training data..'
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
        # Training data is in examples x features
        trainX = trainX[:, 2:]
        feats = feats[2:]

        # Output features and create feature map.
        create_feature_map(settings.OUTDIR + settings.FEATURE_NAMES_FILE, feats)

        with open(settings.OUTDIR + out_tag_name + '.pickle', 'w') as f:
            pickle.dump([trainY, statnums, longs, lats, elevs, date, meth,
                         debug], f)
        pd.DataFrame(trainX, columns=feats).to_csv(settings.OUTDIR
                                                   + out_tag_name + '.csv',
                                                   index=False)
    statnums_x = statnums.copy()
    # Do the same for the testing data.
    if os.path.isfile(settings.OUTDIR + out_tag_name + '_test.pickle'):
        print 'Loading up previously assembled testing data...'
        testX = pd_chunk_csv(settings.OUTDIR + out_tag_name + '_test.csv',
                             chunksize=10000)
    else:
        print 'Assembling the weather testing data..'
        testX, feats = get_gefs_features(model_num, nclose, test_gefs_files,
                                         station_info, method=meth, debug=debug)
        statnums = testX[:, 0]
        elevs = testX[:, 1]
        longs = testX[:, 2]
        lats = testX[:, 3]
        testX = testX[:, 2:]
        date = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV,
                           usecols=(0,)).values.flatten()
        feats = feats[2:]

        with open(settings.OUTDIR + out_tag_name + '_test.pickle', 'w') as f:
            pickle.dump([statnums, longs, lats, elevs, date, meth, debug], f)
        pd.DataFrame(testX, columns=feats).to_csv(settings.OUTDIR + out_tag_name
                                                  + '_test.csv', index=False)

    # Now remove all the data where the pyranometer has failed to make
    # an appropriate measurement.
    trainX, trainY = remove_bad_data(trainX, trainY, statnums_x)
    if settings.PLOTRAWDATA:
        plot_features_vs_targets(trainX, trainY, 'raw_')

        indexes = [2, 17, 22, 57, 77, 32]
        feats = ['Day of Year', 'Forecasted Downward Shortwave Flux',
                 'Forecasted Precipitable Water Over Entire Depth of Atmosphere',
                 'Forecasted Surface Temperature',
                 'Forecasted Upward Shortwave Flux',
                 'Forecasted Specific Humidity']

        w = np.where(statnums_x == 1)[0]
        plot_features_vs_targets_six(trainX[w, :], trainY[w], indexes, feats)

    return trainX, trainY, testX
