import numpy as np
import pickle
import settings

def normalize_data(trainX, trainY, testX):
    """
    Current method of training data
    :param trainX: training X data - weather prediction data + location
    data for each station.
    :param trainY: solar power output from each station
    :return: normalized training + testing data
    """

    x_coeff = np.vstack((trainX, testX)).max(axis=0)
    trainX = trainX / x_coeff

    # Normalize the training data in the same way as the testing data.
    testX = testX / x_coeff

    y_coeff = np.float64(trainY).max(axis=0)
    trainY = trainY / y_coeff

    # Save the normed coefficients
    with open(settings.OUTDIR + settings.NORMCOEFFS_FILE, 'w') as f:
        pickle.dump([x_coeff, y_coeff], f)
    return trainX, trainY, testX, x_coeff, y_coeff
