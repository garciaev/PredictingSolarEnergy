# 21 Questions with the Weather: A Decision Tree Approach to Predicting Day-Ahead Solar Energy

Scaling up solar energy production is key to ending the current reliance on climate-damaging fossil fuels. However, solar energy is variable due 
to clouds and weather. 

Utility companies need accurate forecasts of solar energy availability in order 
to plan the correct mix of fossil fuels and renewable energy to be provided to 
customers. Errors in forecasting solar energy availability could lead to large expenses in extra fossil fuel consumption or emergency purchases of electricity from neighboring utilities. 

Therefore a multi-million dollar problem is forecasting solar energy using numerical weather prediction models as accurately as possible, for precise energy generation planning. 

# The Data
For this project, I used 1.4 GB of weather data (500,000 examples, 80+ features) from 1994-2012 from NOAA/ESRL Global Ensemble Forecast System, obtained from the Kaggle [AMS 2013-2014 Solar Energy Prediction Contest](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest)

These weather forecasts are from 11 different global weather models for 12, 15, 18, 21 and 24 hours ahead at 144 different latitude/longitude locations across 
Oklahoma. 

I used these weather forecasts to predict the total integrated solar energy from sun rise to sun set as measured by 98 Oklahoma [Mesonet](https://www.mesonet.org/) sites using pyranometers from 1994-2012

# The Solution

I implemented the gradient-boosted decision tree model [XGBoost](https://github.com/dmlc/xgboost)
I ran XGBoost to find patterns that link weather forecasts to the actual integrated solar energy measured at each Oklahoma Mesonet site. I was able to predict the actual solar energy available to <6.5% accuracy for the vast majority of days, using a portion of the data not used in model training. 

# Installation 
1. Download the weather forecast files `gefs_test.zip` (or `gefs_test.tar.gz`) and `gefs_train.zip` (or `gefs_train.tar.gz`) from Kaggle data [webpage](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/data)

2. Move the files to the ``Data/`` directory. 
  * If you downloaded the ``.tar.gz`` files:
  ```
  mv gefs_train.tar.gz Data/
  mv gefs_test.tar.gz Data/ 
  ``` 
  * or if you downloaded the ``.zip`` files:
  ```
  mv gefs_train.zip Data/
  mv gefs_test.zip Data/ 
  ```

3. Move into the Data directory: ``cd Data/``

4. Open the ``.tar.gz`` or ``.zip`` files to make the ``Data/train/`` and ``Data/test/`` directories. 
  * If you downloaded the ``.tar.gz`` files, on OSX you can run:
  ```
  tar -xzvf gefs_train.tar.gz
  tar -xzvf gefs_test.tar.gz
  ``` 
  * Or if you downloaded the ``.zip`` files: 
  ```
  unzip gefs_train.zip
  unzip gefs_test.zip
  ``` 
There should now be ``Data/train/`` and ``Data/test/`` directories with multiple ``*.nc`` files such as ``test/apcp_sfc_latlon_subset_20080101_20121130.nc``, ``test/dlwrf_sfc_latlon_subset_20080101_20121130.nc`` etc. Each file gives weather features described on the Kaggle [data webpage](https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest/data)

5. Install the [netCDF4](http://unidata.github.io/netcdf4-python/) module. 

6. Switch back to the ``PredictingSolarEnergy/`` directory: ``cd ..``
# Usage

The code is designed to allow for a lot of experimentation to find the optimal amount of spatial averaging of the weather forecast grid points. 
From ``PredictingSolarEnergy/`` directory, run the code as: 
```
python Code/train_solar_predict.py --outdir OUTDIR --modelnum MODELNUM --numclosegrid NUM --debug DEBUG --method METH --numrandstate NUMRAND --tag TAG 
``` 
* OUTDIR is the user-specified name of the directory for the output files
* MODELNUM is the global weather forecast model to use (integer ``0``-``10``)
* NUM is the number of grid points over which to spatially average a global weather forecast model. Set to ``7``for best results. 
* DEBUG is for debugging, always set to ``0`` (``1`` for debug). 
* METH is a string: ``avg`` for a straightforward spatial average of the forecast models;  ``use4`` for no averaging; and ``wavg`` for using a spatial average weighted by the distance from each weather model grid point to each Mesonet weather station in Oklahoma. 
* NUMRAND is the integer number of times to run XGBoost at different random states. 
* TAG is a string to tag the output files with.  