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
import settings

def normalize_data(trainX, trainY, testX, out_dir):
    """
    Current method of training data
    :param trainX: training X data - weather prediction data + location
    data for each station.
    :param trainY: solar power output from each station
    :return: normalized training + testing data
    """

    x_coeff = np.vstack((trainX,testX)).max(axis=0)
    trainX = trainX / x_coeff

    # Normalize the training data in the same way as the testing data.
    testX = testX / x_coeff

    y_coeff = np.float64(trainY).max(axis=0)
    trainY = trainY / y_coeff

    with open(out_dir + settings.NORMCOEFFS_FILE, 'w') as f:
        pickle.dump([x_coeff, y_coeff], f)
    return trainX, trainY, testX, x_coeff, y_coeff
