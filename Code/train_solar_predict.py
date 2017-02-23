import glob
import os
import argparse
import numpy as np
import pandas as pd
import settings
from assemble_data import assemble_data
from normalize_data import normalize_data
from fit_data import fit_data


if __name__ == "__main__":

    # --------------------------------------
    # Set user parameters from command line
    # ---------------------------------------
    parser = argparse.ArgumentParser()
    argsall = ['--outdir', '--modelnum', '--numclosegrid', '--debug',
               '--method', '--numrandstate', '--tag']
    for ar in argsall:
        parser.add_argument(ar)
    args = parser.parse_args()

    settings.OUTDIR = settings.ROOT_DIR + str(args.outdir) + '/'

    # Load in the GEFS files to be used
    train_gefs_files = np.sort(glob.glob(settings.DATA_DIR_TRAIN + '*nc'))
    test_gefs_files = np.sort(glob.glob(settings.DATA_DIR_TEST + '*nc'))

    # Load in station info
    station_info = pd.read_csv(settings.STATION_INFO_CSV)

    if os.path.isdir(settings.OUTDIR) is False:
        print 'Making output directory ' + settings.OUTDIR
        os.mkdir(settings.OUTDIR)
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
    num_rand = np.int(args.numrandstate)
    print 'Number of random states = ' + str(num_rand)

    # Name of files that will be saved
    tag_name = args.tag

    # ---------------
    # DATA PIPELINE.
    # ----------------
    # 1. Get the .NC files which contain the weather forecasts from
    # 11 different global numerical weather prediction models. Average
    # the data spatially.
    print "Make or grab the training data, averaging spatially..."
    trainX, trainY, testX = assemble_data(tag_name, meth, debug, nclose,
                                          station_info, model_num,
                                          train_gefs_files, test_gefs_files)
    print 'Training sizes:'
    print trainX.shape, trainY.shape

    # 2. Normalize the features, use both train and testing
    # data for the feature normalization to fully encompass the
    # range of X values.
    print "Normalizing data..."
    trainX, trainY, testX, xcoeff, ycoeff = \
        normalize_data(trainX, trainY, testX)
    # 3. Fit the data. Here we use our supervised learning model
    # to find statistically significant correlations between the features
    # (the 12, 15, 18, 21, 24 hours ahead weather forecast) and the prediction
    # variable (the actual total solar energy produced at a given Mesonet
    # station).
    print "Fitting data..."
    fit_data(trainX, trainY, num_rand, ycoeff, testX, tag_name, True)
    print 'Finished.'
