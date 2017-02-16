import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_feat_importance(gbm, modelname):
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
    plt.gcf().savefig(modelname + '_rel.png')

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False,
            figsize=(10, 16))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Absolute importance')
    plt.gcf().savefig(modelname + '_ab.png')


def plot_data_and_residuals(targets, predictions, doy, out_file):
    plt.clf()
    plt.subplot(211)
    plt.scatter(doy, targets, facecolors='none', edgecolors='k')
    plt.scatter(doy, predictions, facecolors='none', edgecolors='r')

    w = np.where(np.abs(targets - predictions) > 0.07)[0]
    plt.scatter(doy[w], targets[w], marker='+', facecolor='b')
    plt.scatter(doy[w], predictions[w], marker='+', facecolor='g')
    plt.xlim([0, 365])
    plt.xlabel('Day of Year')
    plt.ylabel('Normalized Measured Solar Flux')
    plt.title('Model vs Data')
    plt.subplot(212)
    plt.scatter(doy, targets - predictions, facecolors='none', edgecolors='k')
    plt.scatter(doy[w], targets[w] - predictions[w], marker='+', facecolor='b')
    plt.xlabel('Day of Year')
    plt.ylabel('Residuals')
    plt.xlim([0, 365])
    plt.savefig(out_file)
