# Playing 21 Questions with the Weather: A Decision Tree Approach to Predicting Day-Ahead Solar Energy

Scaling up solar energy production is key to ending the current reliance on climate-damaging fossil fuels. However, solar energy is variable due 
to clouds and weather. 

Utility companies need accurate forecasts of solar energy availability in order 
to plan the correct mix of fossil fuels and renewable energy to be provided to 
customers. Errors in forecasting solar energy availability could lead to large expenses in extra fossil fuel consumption or emergency purchases of electricity from neighboring utilities. 

Therefore a multi-million dollar problem is forecasting solar energy using numerical weather prediction models as accurately as possible, for precise energy generation planning. 

# The Data
For this project, I used 1.4 GB of weather data (500,000 examples, 80+ features) from 1994-2012 from NOAA/ESRL Global Ensemble Forecast System, obtained from Kaggle:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest

These weather forecasts are from 11 different global weather models for 12, 15, 18, 21 and 24 hours ahead at 144 different latitude/longitude locations across 
Oklahoma. 

I used these weather forecasts to predict the total integrated solar energy from sun rise to sun set as measured by 98 Oklahoma Mesonet sites using pyranometers from 1994-2012: https://www.mesonet.org/

# The Solution

I implemented extreme gradient boosted trees XGBoost: https://github.com/dmlc/xgboost
I ran XGBoost to find patterns that link weather forecasts to the actual integrated solar energy measured at each Oklahoma Mesonet site. I was able to predict the actual solar energy available to <6.5% accuracy for the vast majority of days, using a portion of the data not used in model training. 

# Installation 
1. Download the data: https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest
2.  