import numpy as np
def is_leap_year(year):
    """
    Check if year is leap year or not
    :param year:
    :return:
    """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0


def doy(Y, M, D):
    """
    Given year, month, day return day of year
    Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7
    :param Y: year (int)
    :param M: month (int 1-12)
    :param D: day (float)
    :return: day of year.
    """
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N


def get_month(dates):
    """
    Get the month from the date string YYYYMMDD.
    :param dates:
    :return: the month of the year corresponding to each element of dates.
    """
    dates_str = map(np.str, list(dates))
    months = [x[4:6] for x in dates_str]
    return np.array(months, dtype=np.float)


def get_doy(dates):
    """
     Convert date string to day of year.
    :param dates:
    :return: array of day of the year corresponding to each element of dates.
    """
    dates_str = map(np.str, dates)
    return np.array([doy(float(dates_str[i][0:4]),
                         float(dates_str[i][4:6]),
                         float(dates_str[i][6:]))
                     for i in range(dates.shape[0])])


