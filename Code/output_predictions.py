import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import settings


def average_models_and_save(modelfiles, testX, ycoeff, xcoeff, INDIR):
    for k, modelfile in enumerate(modelfiles):
        gbm = xgb.Booster({'nthread': 4})
        gbm.load_model(modelfile)
        ypred = gbm.predict(xgb.DMatrix((testX / xcoeff)))
        ypredout = (ypred * np.float64(ycoeff)).reshape(1796, 98, order='F')
        sampsub = None
        if sampsub is None:
            sampsub = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV)
            for i, cols in enumerate(sampsub.columns.values[1:]):
                sampsub[cols] = ypredout[:, i]
        else:
            for i, cols in enumerate(sampsub.columns.values[1:]):
                sampsub[cols] = sampsub[cols]+ypredout[:, i]
    sampsub.to_csv(INDIR+'submit.csv', index=False)


def output_model(ypred, subname):
    """
    Output a single model to the correct Kaggle submission format.
    :param ypred:
    :param subname:
    :return:
    """
    ypred_out = ypred.reshape(1796, 98, order='F')

    sampsub = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV)
    for i, cols in enumerate(sampsub.columns.values[1:]):
        sampsub[cols] = ypred_out[:, i]

    sampsub.to_csv(subname, index=False)

def linearly_optimized_average(predictions, targets):
    """
    Given a list of predictions in the shape of y_train (or y_valid)
    do a linear regression to predict the target.
    :param outputs:
    :param predictions:
    :return:
    """
    x_train = None
    for y in predictions:
        if x_train is None:
            x_train = y
        else:
            x_train = np.hstack((x_train, y))
    # Now do the fit
    lr = LinearRegression()
    lr.fit(x_train, targets)
    #lr.predict()