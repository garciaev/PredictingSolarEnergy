import numpy as np
import pickle
import settings
from sklearn.preprocessing import MinMaxScaler
from visualize_output import plot_features_vs_targets


def normalize_data(trainX, trainY, testX):
    """
    Current method of training data
    :param trainX: training X data - weather prediction data + location
    data for each station.
    :param trainY: solar power output from each station
    :return: normalized training + testing data
    """

    w_ymax = np.where(trainY != np.max(trainY))[0]
    trainY = trainY[w_ymax]
    trainX = trainX[w_ymax, :]

    #print w_ymax
    #print trainY.shape, trainX.shape

    x_coeff = np.vstack((trainX, testX)).max(axis=0)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(np.vstack((trainX, testX)))

    trainX = min_max_scaler.transform(trainX)
    #trainX = trainX / x_coeff

    # Normalize the training data in the same way as the testing data.
    testX = min_max_scaler.transform(testX)
    #testX = testX / x_coeff

    y_coeff = np.float64(trainY).max(axis=0)
    trainY = trainY / y_coeff

    if settings.PLOTRAWDATA:
        plot_features_vs_targets(trainX, trainY, 'norm_')


    # Save the normed coefficients
    with open(settings.OUTDIR + settings.NORMCOEFFS_FILE, 'w') as f:
        pickle.dump([x_coeff, y_coeff], f)
    return trainX, trainY, testX, x_coeff, y_coeff
