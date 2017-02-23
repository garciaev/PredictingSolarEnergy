import os
import numpy as np
import netCDF4 as nc
import pandas as pd
from geopy.distance import vincenty
from date_functions import get_doy
import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def remove_bad_data(trainX, trainY, statnums):
    """
    Now search for and purge equipment failures. When the pyranometer
    fails at a given site, it will be set to a default solar energy value
    that is unique to each site. We can throw this data out, so as not
    to confuse our machine learning algorithm!
    :param trainX:
    :param trainY:
    :param statnums:
    :return:
    """
    bad_data = None
    for i in np.unique(statnums):
        w = np.where(statnums == i)[0]
        uniq, uniq_index, uniq_cnts = np.unique(trainY[w], return_counts=True,
                                                return_index=True)
        bad_uniques = np.where(uniq_cnts > 10)[0]
        bad_data_stat = None
        if len(bad_uniques) == 1:
            # Get the index of the bad data
            bad_data_stat = w[np.where(trainY[w] == uniq[bad_uniques])[0]]
            if bad_data is None:
                bad_data = bad_data_stat
            else:
                bad_data = np.concatenate((bad_data, bad_data_stat))

        if settings.PLOTRAWDATA:
            plt.clf()
            plt.plot(trainX[w, 2], trainY[w], 'k.', markersize=0.1)
            if bad_data_stat is not None:
                plt.plot(trainX[bad_data_stat, 2], trainY[bad_data_stat], 'r.')
            plt.xlabel('Day of Year')
            plt.ylabel('Total Solar Energy')
            plt.savefig(settings.OUTDIR + 'stat_raw_' + str(np.int(i)) + '.png')
    # Now delete the bad data that was compiled in a an array "bad_data"
    trainX = np.delete(trainX, bad_data, axis=0)
    trainY = np.delete(trainY, bad_data)
    return trainX, trainY


def get_max_doy(doy, feature):
    """
    Get the average value of the top 5% of a weather feature
    as function of the day of the year. For example, solar energy output
    will be at maximum during a clear sky. Thus, using this function, we can
    compute a "clear sky index"=(downward radiative flux at given day of year)/
    (maximum downward radiative flux at given day of year).
    :param doy: day of year feature.
    :param feature: weather feature such as downward shortwave radiative flux,
    downward long wave radiative flux, etc.
    :return:
    """
    out_feature = np.zeros(feature.shape[0])
    for day in range(1, 367):
        daysid = np.where(np.abs(doy - day) < 1.0)[0]

        maxday = np.max(np.array(feature[daysid]))
        idgood = np.where(feature[daysid] > 0.95 * maxday)[0]

        out_feature[daysid] = np.mean(feature[daysid[idgood]])
    return out_feature


def mad_percent(predictions, targets):
    """
    Return the median absolute deviation.
    :param predictions:
    :param targets:
    :return:
    """
    return np.median(np.abs(predictions - targets) / targets) * 100.00


def mae_percent(predictions, targets):
    """
    Mean absolute error (MAE).
    The objective function for both XGBoost
    and the Neural Networks to compare
    predicted solar energy to actual solar energy measured.
    :param predictions:
    :param targets:
    :return: mean absolute error
    """
    return np.mean(np.abs(predictions - targets) / targets) * 100.00


def mae(predictions, targets):
    """
    Mean absolute error (MAE).
    The objective function for both XGBoost
    and the Neural Networks to compare
    predicted solar energy to actual solar energy measured.
    :param predictions:
    :param targets:
    :return: mean absolute error
    """
    return np.mean(np.abs(predictions - targets))


def pd_chunk_csv(fname, chunksize):
    """
    Its expensive to purchase a larger memory computer on AWS. This function
    reads in a large data file by loops over chunks.
    :param fname: input .csv file
    :param chunksize: size of data chunk to loop over.
    :return:
    """
    iter_csv = pd.read_csv(fname, dtype=np.float32, engine='c', iterator=True,
                           chunksize=chunksize)
    list_of_chunks = [chunk for chunk in iter_csv]
    ret_arr = None
    for i, chk in enumerate(list_of_chunks):
        if ret_arr is None:
            ret_arr = chk.values
        else:
            ret_arr = np.concatenate((ret_arr, chk.values))
    return ret_arr


def load_gefs(input_file):
    """
    Load in a single weather data GEFs file.
    these are global weather models computed on a grid spaced
    at 50 km. The dimensions of the data array are day,
    weather forecast model, hour, longitude, latitude.
    input_file is the GEFs file. Requires netCDF4 module
    :param input_file: input .nc file
    :return:  dates, latitudes, longitudes, weather data, variable names
    """

    npzname = input_file.split('/')[-1].strip('.nc')+'.npz'
    out_dir = "/".join(input_file.split('/')[:-1])+'/'

    if os.path.isfile(out_dir + npzname):
        outnpz = np.load(out_dir + npzname)
        dates = outnpz['dates']
        lat = outnpz['lat']
        lon = outnpz['lon']
        data = outnpz['data']
        vars = [i for i in outnpz['vars']]
    else:
        din = nc.Dataset(input_file)
        var_name, data = din.variables.popitem()
        var_name = input_file.split('latlon')[0].split('/')[-1][0:-1]
        vars = [var_name + i for i in ['_12', '_15', '_18', '_21', '_24']]
        data = np.array(data)
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 9*16)
        # Get the latitude and longitudes
        lat, lon = np.meshgrid(din.variables['lat'][:], din.variables['lon'][:])
        lat, lon = lat.transpose().flatten(), lon.transpose().flatten()
        # Subtract 360 from the longitude since its in degrees from prime
        # meridian
        lon = lon - 360.000
        # Get the dates
        dates = din.variables['intTime'][:]/100

        np.savez(out_dir + npzname,
                 dates=dates, lat=lat, lon=lon, data=data, vars=vars)

    return dates, lat, lon, data, vars


def get_closest(station_lat, station_lon, lats, lons, ngrid):
    """
    Given a latitude and longitude for a given
    solar station, get the closest weather ngrid points based
    on the latitude and longitude of the weather grid point,
    using the vicenty distance (spheroidal approximation for
    shape of the Earth.
    :param station_lat: solar energy measuring station latitude
    :param station_lon: solar energy measuring station longitude
    :param lats: latitudes of global weather model grid points
    :param lons: longitudes of global weather model grid points
    :param ngrid: number of closest grid points to use
    :return: closest distances, indexes, latitude differences,
    longitude differences
    """
    dists = np.zeros(lats.shape[0])
    lat_dists = np.zeros(lats.shape[0])
    lon_dists = np.zeros(lats.shape[0])
    for i in range(lats.shape[0]):
        dists[i] = vincenty((station_lat, station_lon),
                            (lats[i], lons[i])).meters
        lat_dists[i] = lats[i] - station_lat
        lon_dists[i] = lons[i] - station_lon
    # Now get ngrid closest. Typically, ngrid = 4
    # Get the returned dists, lats, and lons
    ind = np.argsort(dists)[0:ngrid]
    dists_close = dists[ind]
    lat_dists_close = lat_dists[ind]
    lon_dists_close = lon_dists[ind]
    return dists_close, ind, lat_dists_close, lon_dists_close


def create_feature_map(fileout, features):
    """
    Output a file with all the weather and distance feature names.
    :param fileout: file to print list of features to
    :param features: list of features
    :return: nothin!
    """
    outfile = open(fileout, 'w')
    i = 0
    outfile.write('{0}\t{1}\n'.format('FeatureNum', 'Feature'))
    for feat in features:
        outfile.write('{0}\t{1}\n'.format(i, feat))
        i = i + 1
    outfile.close()
    return None


def add_feature(xdata, val, numfeats):
    """
    Add features (columns) to the training data.
    :param xdata: the actual input data
    :param val: number to add as a column
    :param numfeats: number of times to replicate val
    :return: xdata with column added in front.
    """
    feat = np.zeros(numfeats)
    feat[:] = val
    return np.hstack((feat.reshape(numfeats, 1), xdata))


def get_gefs_features(model_num, num_closest_grid, ncfiles,
                      station_info, method, debug=False):
    """
    This function loops over every weather station, and weather feature, grabs
    the closest weather grid points for a given global model,
    and outputs an array that serves as the training data.
    :param model_num: which global weather model to use (int 0-10)
    :param num_closest_grid: the number of nearest weather forecast
    model grid points.
    :param ncfiles: the list of .nc files to use
    :param station_info: information about each solar energy measurement station
    :param method: use4, wavg, avg - average over the weather
    grid points or not?
    :param debug: run a limited number of stations, just to debug output.
    :return: features, feature_names
    """

    # Just to get latitudes and longitudes...
    _, grid_lat, grid_lon, trainsdat, _ = load_gefs(ncfiles[1])
    nfeat = trainsdat.shape[0]
    if method != 'use4':
        feats = ['stat_num', 'elev', 'long', 'lat', 'doy']
    if method == 'use4':
        dstr = ['d1_lat', 'd2_lat', 'd3_lat', 'd4_lat',
                'd1_lon', 'd2_lon', 'd3_lon', 'd4_lon',
                'd1', 'd2', 'd3', 'd4', 'doy']
        feats = ['stat_num', 'elev', 'long', 'lat'] + dstr

    # column 0 is station number, col1=elev,col2=long,col3=lat,col4=month
    # other features are mentioned above in the "dstr" variable.
    # Loop over every station getting 4 closest weather grid points
    close = []
    dists = []
    lat_dists = []
    lon_dists = []
    stat_lats = station_info['nlat'].values
    stat_lons = station_info['elon'].values
    stat_elev = station_info['elev'].values

    for ii in range(station_info.shape[0]):
        dist, indexes, lat_dist, lon_dist = get_closest(
            stat_lats[ii],
            stat_lons[ii],
            grid_lat, grid_lon,
            num_closest_grid
        )
        dists.append(dist)
        close.append(indexes)
        lat_dists.append(lat_dist)
        lon_dists.append(lon_dist)
    # Now interpolate the weather variable to a specific longitude and latitude
    # using the 4 nearest points.
    # Loop over every station, each station has 4 closest weather grid points
    if debug:
        rng = range(2)
    else:
        rng = range(len(close))

    X_train = None
    # Loop over every station
    for ii in rng:
        print 'Station ' + str(ii)
        # grab all the weather variables for a station
        stat_x = None
        # Loop over all the weather variables for a station
        for h, ncfile_cur in enumerate(ncfiles):

            dates, _, _, stat_weather, feats_cur = load_gefs(ncfile_cur)
            # Do the 4 averaging of nearest weather stations. NO WEIGHTING
            if method == 'avg':
                stat_weather = np.mean(
                    stat_weather[:, model_num, :, close[ii]], axis=0)
                if X_train is None:
                    feats = feats + feats_cur
            # WEIGHT BY THE DISTANCE.
            if method == 'wavg':
                wts = np.array(dists[ii])
                stat_weather = np.average(
                    stat_weather[:, model_num, :, close[ii]],
                    axis=0, weights=wts)
                if X_train is None:
                    feats = feats + feats_cur
            # If use 4 distances
            if method == 'use4':
                # Use the 4 closest distances as features
                dists_curs = dists[ii]
                # Use the 4 closest latitudes and longitudes as features
                lat_cur = lat_dists[ii]
                lon_cur = lon_dists[ii]
                # Grab all dates, specific model number, 4 closest grid points
                stat_weather = stat_weather[:, model_num, :, close[ii]]

                # Reshape the training data appropriately by stacking
                # grid points on top of eachother
                stat_weather = stat_weather.reshape(
                    stat_weather.shape[1],
                    stat_weather.shape[0] * stat_weather.shape[2]
                )
                if X_train is None:
                    feat_mod = ['_gridpt' + str(hh) for hh in
                                range(num_closest_grid)]
                    feats = feats + [f_ + fm_ for fm_ in feat_mod
                                     for f_ in feats_cur]
            # Add columns to front of array, only once.
            if stat_x is None:
                stat_x = np.hstack(
                    (get_doy(dates).reshape(dates.shape[0], 1),
                     stat_weather)
                )

                # If use 4 add distances add these 4 features out front
                if method == 'use4':
                    for dists_curs_ in dists_curs:
                        stat_x = add_feature(stat_x, dists_curs_, nfeat)
                    for lat_cur_ in lat_cur:
                        stat_x = add_feature(stat_x, lat_cur_, nfeat)
                    for lon_cur_ in lon_cur:
                        stat_x = add_feature(stat_x, lon_cur_, nfeat)
                # add stat latitude
                stat_x = add_feature(stat_x, stat_lats[ii], nfeat)
                # add stat longitude
                stat_x = add_feature(stat_x, stat_lons[ii], nfeat)
                # add stat elevation
                stat_x = add_feature(stat_x, stat_elev[ii], nfeat)
                # add stat number
                stat_x = add_feature(stat_x, ii+1, nfeat)
            else:
                # Keep adding on those weather variables horizontally
                stat_x = np.hstack((stat_x, stat_weather))
        # Now stack stations on top of each other
        if X_train is None:
            X_train = stat_x
        else:
            X_train = np.vstack((X_train, stat_x))

    # Add additional features!
    maxflux_daily = get_max_doy(X_train[:, 4],
                                np.sum(X_train[:, 15:20], axis=1))
    for hh in range(15, 20):
        addfeat = X_train[:, hh] / maxflux_daily
        X_train = np.hstack((X_train, addfeat.reshape(X_train.shape[0], 1)))

    feats = feats + [fe + '_daynorm' for fe in feats[15:20]]

    maxflux_daily = get_max_doy(X_train[:, 4],
                                np.sum(X_train[:, 25:30], axis=1))
    for hh in range(25, 30):
        addfeat = X_train[:, hh] / maxflux_daily
        X_train = np.hstack((X_train, addfeat.reshape(X_train.shape[0], 1)))
    feats = feats + [fe + '_daynorm' for fe in feats[25:30]]

    return X_train, feats
