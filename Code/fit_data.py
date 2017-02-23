import random
import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from grid_search_xgboost import grid_search_xgboost
import settings


def fit_data(trainX, trainY, num_rand, ycoeff, testX, tag_name, visualize):
    """
    Fit the weather prediction data using either XGBoost (decision trees)
    Neural networks, or both.
    :param trainX:
    :param trainY:
    :param num_rand:
    :param ycoeff:
    :param testX:
    :param tag_name:
    :return:
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
    """

    parameters = {"eta": settings.ETA,
                  "alpha": settings.ALPHA,
                  "lambda": settings.LAMBDA,
                  "gamma": settings.GAMMA,
                  "colsample_bytree": settings.COLSAMPLE_BYTREE,
                  "colsample_bylevel": settings.COLSAMPLE_BYLEVEL,
                  "max_delta_step": settings.MAX_DELTA_STEP,
                  "min_child_weight": settings.MIN_CHILD_WEIGHT,
                  "scale_pos_weight": settings.SCALE_POS_WEIGHT,
                  "subsample": settings.SUBSAMPLE,
                  "max_depth": settings.MAX_DEPTH}
    early_stop = settings.EARLY_STOP
    n_steps = settings.N_STEPS
    objective = settings.OBJECTIVE
    booster = settings.BOOSTER
    eval_metric = settings.EVAL_METRIC
    feats = pd.read_csv(settings.OUTDIR + settings.FEATURE_NAMES_FILE,
                        delimiter='\t')['Feature'].values

    out_dir = settings.OUTDIR
    train_frac = settings.TRAIN_FRAC

    n_feats = trainX.shape[0]
    indices = np.arange(n_feats)
    # Split between train and evaluation data
    rands = [random.randint(0, 100000) for i in range(num_rand)]
    # Loop over multiple random seeds
    for jj, rnd in enumerate(rands):
        id_train, id_valid = train_test_split(indices, train_size=train_frac,
                                              random_state=rnd)
        print 'Running XGBoost grid...'
        print trainX[id_train].shape
        grid_search_xgboost(trainX[id_train], trainY[id_train],
                            trainX[id_valid], trainY[id_valid],
                            parameters, n_steps, early_stop,
                            objective, booster, eval_metric,
                            rnd, out_dir, ycoeff, testX, feats,
                            tag_name, visualize)

        model_tag = tag_name + '_ids_random_seed' + str(rnd)
        # Save the data for this run of the model.
        with open(out_dir + model_tag + '.pickle', 'w') as f:
            pickle.dump([tag_name, model_tag, rnd, indices, id_train, id_valid,
                         feats, jj], f)
