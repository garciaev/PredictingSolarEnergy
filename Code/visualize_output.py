import numpy as np
import pandas as pd
import matplotlib
import operator
import settings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn
from matplotlib.cbook import get_sample_data


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
        plt.ylabel('Measured Total Solar Energy $J$ meter$^{-2}$')
        plt.savefig(settings.OUTDIR + tag + feat + '_vs_target.png')


def plot_features_vs_targets_six(data, target, indexes, feats):
    fig, axs = plt.subplots(3, 2, figsize=(16, 16))
    fig.subplots_adjust(hspace=.13, wspace=.1)
    matplotlib.rcParams['font.weight'] = 900
    matplotlib.rcParams['axes.linewidth'] = 10
    axs = axs.ravel()

    for j, feat in enumerate(feats):
        axs[j].scatter(data[:, indexes[j]], (target - np.min(target))
                       / np.max(target), marker='.',
                       edgecolors='k', facecolors='none',
                       linewidths=3.0)
        axs[j].set_xlabel(feat, fontweight='black', size='xx-large')
        axs[j].set_ylabel('Actual Solar Energy', fontweight='black', size=28)
        axs[j].xaxis.set_tick_params(labelsize=16, width=2)
        axs[j].yaxis.set_tick_params(labelsize=16, width=2)
        for axis in ['top', 'bottom', 'left', 'right']:
            axs[j].spines[axis].set_linewidth(2)

    plt.tight_layout()

    plt.savefig(settings.OUTDIR + 'six_vs_target.png')


def plot_clear_sky_index_residuals(targets, predictions, clear_sky_index,
                                   out_file, nbins):
    percent_diff = (targets - predictions) #/ targets
    w1 = np.where(percent_diff > 50.0)[0]
    w2 = np.where(percent_diff < -50.0)[0]
    w3 = np.where(clear_sky_index < 0.05)[0]

    percent_diff_plot1 = np.delete(percent_diff, np.hstack((w1, w2, w3)))
    clear_sky_index_plot1 = np.delete(clear_sky_index, np.hstack((w1, w2, w3)))

    hist, bin_edges = np.histogram(clear_sky_index_plot1, bins=nbins)
    hist_ind = np.digitize(clear_sky_index_plot1, bin_edges)

    percent_diff_plot = np.zeros(nbins)
    clear_sky_index_plot = np.zeros(nbins)
    for jj in range(nbins):
        in1 = np.where(hist_ind == jj)[0]
        #print hist_ind[in1]
        percent_diff_plot[jj] = np.mean(np.abs(percent_diff_plot1[in1]))
        clear_sky_index_plot[jj] = np.mean(np.abs(clear_sky_index_plot1[in1]))
        print clear_sky_index_plot[jj], percent_diff_plot[jj]

    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(12, 10))

    axs.scatter(clear_sky_index_plot, percent_diff_plot, marker='.', s=200,
                edgecolors='k', facecolors='k', linewidths=3.0)
    axs.plot(clear_sky_index_plot, percent_diff_plot, 'k--', linewidth=3.0)

    axs.set_xlabel('Clear Sky Index', fontweight='black', size=32)
    axs.set_ylabel('Mean Absolute Error', fontweight='black', size=28)

    axs.xaxis.set_tick_params(labelsize=20, width=2)
    axs.yaxis.set_tick_params(labelsize=20, width=2)

    axs.set_xlim([0.0, 1.07])
    axs.set_ylim([0.01, 0.125])
    axs.minorticks_on()
    axs.tick_params('both', length=1, width=2, which='major')
    axs.tick_params('both', length=0.5, width=1, which='minor')

    for axis in ['top', 'bottom', 'left', 'right']:
        axs.spines[axis].set_linewidth(5)
    axs.text(0.37, 0.11, 'Partially Cloudy Days \n Semi-Unpredictable',
             fontweight='black', fontsize=16)
    axs.text(0.05, 0.11, '100% Cloudy Days \nPredictable',
             fontweight='black', fontsize=16)
    axs.text(0.78, 0.11, 'Clear Skies \nPredictable',
             fontweight='black', fontsize=16)
    axs.set_title('Predictability of Sun Light Available \n For Solar Panels'
                  ' 24 hours ahead',
                  fontsize=28, fontweight='black')

    im = plt.imread('verycloudy.jpeg')
    newax = fig.add_axes([0.15, 0.56, 0.2, 0.2], anchor='NE', zorder=1)
    newax.imshow(im)
    newax.axis('off')

    im = plt.imread('partially_cloudy.jpeg')
    newax = fig.add_axes([0.4, 0.56, 0.2, 0.2], anchor='NE', zorder=1)
    newax.imshow(im)
    newax.axis('off')

    im = plt.imread('clearsky.jpeg')
    newax = fig.add_axes([0.65, 0.56, 0.2, 0.2], anchor='NE', zorder=1)
    newax.imshow(im)
    newax.axis('off')

    plt.savefig(out_file)


def plot_hist_residuals(targets, predictions, out_file):
    percent_diff = 100 * (targets - predictions) / targets
    w1 = np.where(percent_diff > 50.0)[0]
    w2 = np.where(percent_diff < -50.0)[0]

    percent_diff_plot = np.delete(percent_diff, np.hstack((w1, w2)))
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))

    axs.hist(percent_diff_plot, normed=True, bins=100,
             facecolor='white', edgecolor='black', linewidth=2)
    axs.set_xlabel('Predicted Solar Energy - Actual %',
               fontweight='black', size=34)
    axs.set_ylabel('Frequency', fontweight='black', size=34)
    axs.xaxis.set_tick_params(labelsize=20, width=2)
    axs.yaxis.set_tick_params(labelsize=20, width=2)

    axs.minorticks_on()
    axs.tick_params('both', length=30, width=4, which='major')
    axs.tick_params('both', length=15, width=3, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        axs.spines[axis].set_linewidth(5)

    plt.savefig(out_file)


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
