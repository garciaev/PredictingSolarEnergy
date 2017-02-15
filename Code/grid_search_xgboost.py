import sys
import xgboost as xgb
import pickle
import numpy as np
from preprocess_global_weather_data import mae
from output_predictions import output_model
from output_predictions import plot_feat_importance

def grid_search_xgboost(X_train, y_train, X_valid, y_valid, params, nsteps,
                        early_stop, objective, booster, eval_metric,
                        random_seed, outdir, y_coeff, testX):
    """
    Make my own XGBoost grid searcher - sci-kit learn one is awful!
    params is a dictionary of lists that could like this:
    params = {"eta": [0.1],
                  "alpha": [0.0],
                  "lambda": [0.0],
                  "gamma": [0.0],
                  "colsample_bytree": [1.0],
                  "colsample_bylevel": [1.0],
                  "max_delta_step": [0.0],
                  "min_child_weight": [0, 1.0, 2.0, 3.0, 5.0, 6.0],
                  "scale_pos_weight": [0.0],
                  "subsample": [1.0],
                  "max_depth": [5.0, 10, 15, 20, 23, 30, 40, 50]}
    :param X_train: training data
    :param y_train: training data
    :param X_valid: validation data
    :param y_valid: validation data
    :param params: discussed above.
    :param nsteps: number of steps to run XGBoost
    :param early_stop: the early stopping parameter for XGBoost
    :param objective: objective function to use
    :param booster: always gbtree, could be gblinear
    :param eval_metric: the evaluation metric to watch
    :param random_seed: the random initialization for XGBoost
    :param outdir: place to put the output of this run of XGBoost
    :return:
    """
    # grid_search xgboost will make a grid of all different permutations,
    # and force into a standard array set up.
    defaults = {"max_depth": [6.0],
                "min_child_weight": [1.0],
                "scale_pos_weight": [0.0],
                "max_delta_step": [0.0],
                "eta": [0.1],
                "alpha": [0.0],
                "lambda": [0.0],
                "gamma": [0.0],
                "colsample_bytree": [1.0],
                "colsample_bylevel": [1.0],
                "subsample": [1.0]}

    # If a parameter value is not in the dictionary, add it.
    for par in defaults.keys():
        if par not in params.keys():
            params[par] = defaults[par]

    nk = len(params.keys())
    gridvals = np.array(np.meshgrid(params['eta'],
                                    params['alpha'],
                                    params['lambda'],
                                    params['gamma'],
                                    params['colsample_bytree'],
                                    params['colsample_bylevel'],
                                    params['max_delta_step'],
                                    params['min_child_weight'],
                                    params['scale_pos_weight'],
                                    params['subsample'],
                                    params['max_depth'])).T.reshape(-1, nk)
    npts = gridvals.size
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)
    gridpt_number = 0
    # Loop over different grid values
    for eta, al, lm, gm, colt, coll, md, mn, sc, ss, mx in gridvals:
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        params_cur = {"objective": objective,
                      "booster": booster,
                      "eval_metric": eval_metric,
                      "silent": 0,
                      "eta": eta,
                      "alpha": al,
                      "lambda": lm,
                      "gamma": gm,
                      "colsample_bytree": colt,
                      "colsample_bylevel": coll,
                      "max_delta_step": md,
                      "min_child_weight": mn,
                      "scale_pos_weight": sc,
                      "subsample": ss,
                      "max_depth": np.int(mx),
                      "base_score": np.median(y_train)}
        evals_result = {}  # store to plot learning curves later
        # train xgboost using parameters and X_train, y_train
        gbm = xgb.train(params_cur, dtrain, nsteps, evals=watchlist,
                        verbose_eval=True, early_stopping_rounds=early_stop,
                        evals_result=evals_result)
        print 'Parameters:', params_cur
        print 'Random Seed:', random_seed

        print 'Actual Train MAE:', mae(gbm.predict(xgb.DMatrix(X_train))
                                       * y_coeff, y_train * y_coeff)
        print 'Actual Valid MAE:', mae(gbm.predict(xgb.DMatrix(X_valid))
                                       * y_coeff, y_valid * y_coeff)

        # Save all the ouput of XGBoost
        # Save model in model and text format.
        model_tag = 'random_seed_' + str(random_seed) + \
                    'gridpt_number' + str(gridpt_number)

        gbm.save_model(outdir + model_tag + '.model')

        with open(outdir + 'parms_' + model_tag + '.pickle', 'w') as f:
            pickle.dump([params_cur, objective, random_seed,
                         early_stop, nsteps, evals_result, npts, model_tag], f)


        # Output the model.
        output_model(gbm, testX, outdir + model_tag + '_submit.csv', y_coeff)
        gridpt_number = gridpt_number + 1
