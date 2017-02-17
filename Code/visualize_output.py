import numpy as np
import pandas as pd
import matplotlib
import operator
import settings
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_feat_importance(gbm, model_name):
    importance = gbm.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            figsize=(10, 16))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig(model_name + '_rel.png')

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            figsize=(10, 16))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Absolute importance')
    plt.gcf().savefig(model_name + '_ab.png')


def plot_features_vs_targets(data, target, tag):
    feats = pd.read_csv(settings.OUTDIR + settings.FEATURE_NAMES_FILE,
                        delimiter='\t')['Feature'].values
    for j, feat in enumerate(feats):
        plt.clf()
        plt.plot(data[:, j], target, 'k.', markersize=0.1)
        plt.xlabel(feat)
        plt.ylabel('Total Solar Energy Measured')
        plt.savefig(settings.OUTDIR + tag + feat + '_vs_target.png')


def plot_data_and_residuals(targets, predictions, doy, out_file):
    doy = doy * 365.00
    plt.clf()

    plt.subplot(311)
    plt.scatter(doy, targets, facecolors='none', edgecolors='k')
    plt.scatter(doy, predictions, facecolors='none', edgecolors='r')
    w = np.where(np.abs(targets - predictions) > 0.07)[0]
    plt.scatter(doy[w], targets[w], marker='+', facecolor='b')
    plt.scatter(doy[w], predictions[w], marker='+', facecolor='g')
    plt.xlim([0, 365])
    plt.xlabel('Day of Year')
    plt.ylabel('Normalized Measured Solar Flux')
    plt.title('Model vs Data')

    plt.subplot(312)
    plt.scatter(doy, targets - predictions, facecolors='none', edgecolors='k')
    plt.scatter(doy[w], targets[w] - predictions[w], marker='+', facecolor='b')
    plt.xlabel('Day of Year')
    plt.ylabel('Residuals')
    plt.xlim([0, 365])

    plt.subplot(313)
    plt.scatter(targets, targets - predictions, facecolors='none',
                edgecolors='k')
    plt.xlabel('Actual Solar Energy')
    plt.ylabel('Actual - Predicted')

    plt.savefig(out_file)
