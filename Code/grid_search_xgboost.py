import xgboost as xgb
import pickle
import numpy as np
from preprocess_global_weather_data import mae, mae_percent
from preprocess_global_weather_data import mad_percent
from output_predictions import output_model
from visualize_output import plot_feat_importance, plot_data_and_residuals


def grid_search_xgboost(X_train, y_train, X_valid, y_valid, params, nsteps,
                        early_stop, objective, booster, eval_metric,
                        random_seed, out_dir, y_coeff, testX, feats,
                        tag_name, visualize):
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
    :param X_train:
    :param y_train:
    :param X_valid:
    :param y_valid:
    :param params:
    :param nsteps:
    :param early_stop:
    :param objective:
    :param booster:
    :param eval_metric:
    :param random_seed:
    :param out_dir:
    :param y_coeff:
    :param testX:
    :param feats:
    :param tag_name:
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
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feats)
    dvalid = xgb.DMatrix(X_valid, y_valid, feature_names=feats)
    gridpt = 0
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
        # train XGBoost using parameters and X_train, y_train
        gbm = xgb.train(params_cur, dtrain, nsteps, evals=watchlist,
                        verbose_eval=True, early_stopping_rounds=early_stop,
                        evals_result=evals_result)

        print 'Parameters:', params_cur
        print 'Random Seed:' + str(random_seed)

        y_train_pred = gbm.predict(xgb.DMatrix(X_train, feature_names=feats))
        print 'Actual Train MAE:' \
              + str(mae(y_train_pred * y_coeff, y_train * y_coeff))

        print 'Train MAE (%):' \
              + str(mae_percent(y_train_pred * y_coeff, y_train * y_coeff))

        print 'Train MAD (%):' \
              + str(mad_percent(y_train_pred * y_coeff, y_train * y_coeff))

        y_valid_pred = gbm.predict(xgb.DMatrix(X_valid, feature_names=feats))
        print 'Actual Valid MAE:' \
              + str(mae(y_valid_pred * y_coeff, y_valid * y_coeff))

        print 'Valid MAE (%):' \
              + str(mae_percent(y_valid_pred * y_coeff, y_valid * y_coeff))

        print 'Valid MAD (%):' \
              + str(mad_percent(y_valid_pred * y_coeff, y_valid * y_coeff))

        # Save all the out put of XGBoost
        # Save model in model and text format.
        model_tag = tag_name + '_rnd_' + str(random_seed) \
                    + 'gridpt_' + str(gridpt)

        gbm.save_model(out_dir + model_tag + '.model')

        with open(out_dir + 'parms_' + model_tag + '.pickle', 'w') as f:
            pickle.dump([params_cur, objective, random_seed, early_stop, nsteps,
                         evals_result, npts, model_tag, y_train, y_train_pred,
                         y_valid, y_valid_pred], f)

        # Output the XGBoost model, making it ready for submission to Kaggle.
        # Reverse scale the predictions for Kaggle, since they come from testX,
        # which contains the testing data that is already scaled.
        if testX is not None:
            print 'Output...'
            y_pred = gbm.predict(xgb.DMatrix(testX, feature_names=feats)) \
                     * y_coeff
            output_model(y_pred, out_dir + model_tag + '_submit.csv')

        # Plot the feature importance.
        plot_feat_importance(gbm, out_dir + model_tag)
        gridpt = gridpt + 1
        if visualize:
            # Plot the residuals "model-data" for the training data
            plot_data_and_residuals(y_train, y_train_pred, X_train[:, 2],
                                    out_dir + 'train_' + model_tag
                                    + '_resids.png')
            # Plot the residuals "model-data" for the validation data data
            plot_data_and_residuals(y_valid, y_valid_pred, X_valid[:, 2],
                                    out_dir + 'valid_' + model_tag
                                    + '_resids.png')
