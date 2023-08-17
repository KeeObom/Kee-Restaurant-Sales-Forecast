from datetime import date
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

#  DATA PREPARATION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the csv file
# Read the excel file into a pandas dataframe
# df = pd.read_excel('sales_science.xlsx', engine='openpyxl')
df = pd.read_csv('sales_science _csv.csv')

import datetime as dt

df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%y", dayfirst=True)

# Expand the date
# Now it has been converted to datetime, let's extract the month, week, day of month and day of week from the date
df['Date_month'] = df['Date'].dt.month
#df['Date_week'] = df['Date'].dt.week
df['Date_week'] = df['Date'].dt.isocalendar().week
df['Date_day'] = df['Date'].dt.day
df['Date_dayofweek'] = df['Date'].dt.dayofweek

print(df.Date_day.values)

# Days of week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# duplicate df
df_corr = df

# Move Values column to the last to make Training data easier
column_to_move = df_corr.pop("Value")
df_corr.insert(11, "Value", column_to_move)

#   MACHINE LEARNING REGRESSOR MODEL
# Define X and y
X = df_corr[['Tavg', 'Havg', 'Wavg', 'Pavg', 'Date_day', 'Date_dayofweek']].to_numpy()
y = df_corr['Value'].to_numpy()

# Perform Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_sc = sc.fit_transform(X)

# Splitting the data into train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)

from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from time import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


# import gridsearchcv
from sklearn.model_selection import GridSearchCV

# Define parameters
param_grid = {"n_estimators": [200,250,300], "min_samples_leaf": np.arange(1, 4), "max_features": [0.3, 0.4, 0.5,'sqrt'],"max_samples": np.arange(0.4, 0.7, 0.1)}

# Working with the parameters
grid2 = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=3)
model = grid2.fit(X_train,y_train)
print(model.best_params_,'\n')
print(model.best_estimator_,'\n')

# So after Tuning ang getting the best parameters for the RandomForest, let's use it and fit the the model
rfcv = RandomForestRegressor(max_features=0.5, max_samples=0.6, n_estimators=200)
rfcv.fit(X_train, y_train)

# Load trained model instead of training model each time program runs
from sklearn import model_selection, datasets
import joblib
import pickle

# Save the model using joblib
joblib.dump(rfcv, "random_forest_model.joblib")