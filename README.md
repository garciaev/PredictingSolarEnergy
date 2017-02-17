# Playing 21 Questions with the Weather: A Decision Tree Approach to Predicting Day-Ahead Solar Energy

Scaling up solar energy production is key to ending the current reliance on fossil fuels and limit climate change. However, solar energy is variable due 
to clouds and weather. 

Errors in the forecast could lead to large expenses in extra fossil fuel consumption or emergency purchases of electricity from other companies. Therefore, predicting day-ahead solar energy availability using weather forecasts is critical for utility companies.

# The Data
For this project, I used weather data from 1994-2012 from NOAA/ESRL Global Ensemble Forecast System, obtained from Kaggle:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest
These weather are forecasts for 12, 15, 18, 21 and 24 hours ahead. 

I used these weather data to predict the total integrated solar energy available as measured by 98 Oklahoma Mesonet sites using pyranometers from 1994-2012: https://www.mesonet.org/

# The Solution

I implemented extreme gradient boosted trees XGBoost: https://github.com/dmlc/xgboost
I ran XGBoost to find the statistically relevant patterns that link weather forecasts to the actual integrated solar energy measured at each Oklahoma Mesonet site. I was able to predict the actual solar energy available to within 7% accuracy. 

