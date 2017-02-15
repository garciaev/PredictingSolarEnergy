import xgboost as xgb
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import os
import settings

def plot_feat_importance(gbm, out_dir, modelname):

    importance = gbm.get_fscore(fmap=in_dir + 'test.xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            figsize=(10, 16))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig(modelname + '_rel.png')

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            figsize=(10, 16))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Absolute importance')
    plt.gcf().savefig(modelname + '_ab.png')


def output_model(gbm, testX, subname, ycoeff):

    # Predict the Y data
    ypred = gbm.predict(xgb.DMatrix(testX))

    # Reverse scale the Y data
    ypred = ypred * ycoeff
    ypred_out = ypred.reshape(1796, 98, order='F')

    sampsub = pd.read_csv(settings.SAMPLE_SUBMISSION_CSV)
    for i, cols in enumerate(sampsub.columns.values[1:]):
        sampsub[cols] = ypred_out[:, i]

    sampsub.to_csv(subname, index=False)
    # plot_feat_importance(gbm, out_dir, modelname)