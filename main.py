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
import pandas as pd
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
df['Date_week'] = df['Date'].dt.week
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# import gridsearchcv
from sklearn.model_selection import GridSearchCV

# Define parameters
# param_grid = {"n_estimators": [200,250,300], "min_samples_leaf": np.arange(1, 4), "max_features": [0.3, 0.4, 0.5,'sqrt'],"max_samples": np.arange(0.4, 0.7, 0.1)}

# Working with the parameters
# grid2 = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=3)
# model = grid2.fit(X_train,y_train)
# print(model.best_params_,'\n')
# print(model.best_estimator_,'\n')

# So after Tuning ang getting the best parameters for the RandomForest, let's use it and fit the the model
rfcv = RandomForestRegressor(max_features=0.5, max_samples=0.6, n_estimators=200)
rfcv.fit(X_train, y_train)

# Make prediction
yhat = rfcv.predict(X_test)
# Check metrics
from sklearn.metrics import r2_score

score_rf = r2_score(y_test, yhat)
print("The accuracy of our model is {}%".format(round(score_rf, 2) * 100))

# dataframe of Actual and predicted value
df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': yhat})
# print(df_rf_1)

# Mean absolute error
from sklearn.metrics import mean_absolute_error

score_rf = mean_absolute_error(y_test, yhat)
print("The Mean Absolute Error of our Model is {}".format(round(score_rf, 2)))

# Mean squared Error score
from sklearn.metrics import mean_squared_error
import numpy as np

score_mse = np.sqrt(mean_absolute_error(y_test, yhat))
print("The Root Mean Squared Error of our Model is {}".format(round(score_mse, 2)))

# Get the r-squared score
print('How well model explains trained data: ', rfcv.score(X_train, y_train))

print('The R squared value is: ', r2_score(y_test, yhat))

print("RMSE", np.sqrt(mean_squared_error(y_test, yhat)))

#    THE DASH APP

# Actual vs predicted figure
fig = px.line(df_rf, x=df_rf.index, y=["Actual", "Predicted"], title="Actual vs Predicted Sales", template="seaborn")

# fig = px.line(df_rf, x=df_rf.index, y=["Actual", "Predicted"])
        # return fig

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div(style={"textAlign": "ljust", 'width': "1200px", "font-family": "Verdana"},

                      children=[

                          html.H1(children="Prediction Tool"),

                          html.H2(id="prediction_result", children="Predicted Sales is: "),

                          html.H4(children="Day of month"),

                          dcc.Slider(
                              id="date_day_input",  # change to date_day_input
                              min=df_corr["Date_day"].min(),
                              max=df_corr["Date_day"].max(),
                              step=1,
                              value=0,
                              marks={i: '{}'.format(i) for i in
                                     range(df_corr["Date_day"].min(), df_corr["Date_day"].max() + 1)}
                          ),

                          html.H4(children="Day of the Week"),

                          dcc.Dropdown(
                              id='dayofweek',  # change to dayofweek
                              options=[
                                  {"label": "Monday", "value": 0},
                                  {"label": "Tuesday", "value": 1},
                                  {"label": "Wednesday", "value": 2},
                                  {"label": "Thursday", "value": 3},
                                  {"label": "Friday", "value": 4},
                                  {"label": "Saturday", "value": 5},
                                  {"label": "Sunday", "value": 6}]
                          ),

                          html.H4(children="Temperature"),

                          dcc.Input(
                              id="tavg_input",  # change to Tavg
                              type="number",
                              placeholder="Fill in",
                          ),

                          # html.Div(id='updatemode-output-container_2', style={'margin-top': 20}), #not sure what for

                          html.H4(children="Humidity"),

                          dcc.Input(
                              id="havg_input",  #
                              type="number",
                              placeholder="Fill in",
                          ),

                          html.H4(children="Wind Speed"),

                          dcc.Input(
                              id="wavg_input",
                              type="number",
                              placeholder="Fill in",
                          ),

                          html.H4(children="Pressure"),

                          dcc.Input(
                              id="pavg_input",
                              type="number",
                              placeholder="Fill in",
                          ),

                          html.Div(dcc.Graph(id="Actual_vs_predicted",
                            config={"displayModeBar": False},
                            animate=True,
                            figure=fig)),

                      ])


# CALLBACKS

# Callback for the visualization, lineplot does not need callback



# Callback for the predicted value
@app.callback(Output(component_id="prediction_result", component_property="children"),
              [Input(component_id="date_day_input", component_property="value"),
               Input(component_id="tavg_input", component_property="value"),
               Input(component_id="havg_input", component_property="value"),
               Input(component_id="wavg_input", component_property="value"),
               Input(component_id="pavg_input", component_property="value"),
               Input(component_id="dayofweek", component_property="value")])
def make_prediction(tavg, havg, wavg, pavg, date_day,
                      dayofweek):  # the order Tavg, Havg, Wavg, Pavg, Date_day, Date_dayofweek
    df_prediction1 = pd.DataFrame(
        {'Tavg': tavg, 'Havg': havg, 'Wavg': wavg, 'Pavg': pavg, 'Date_day': date_day, 'Date_dayofweek': dayofweek}, index=[0])
    X_prediction1 = df_prediction1[['Tavg', 'Havg', 'Wavg', 'Pavg', 'Date_day', 'Date_dayofweek']].to_numpy()

    try:
        input_X = sc.transform(X_prediction1)
        # make prediction with the model rfcv
        rf_prediction1 = rfcv.predict(input_X)
        return "Predicted Sales: N{}".format(rf_prediction1)



    except ValueError:
        return "Fill in all input values below"


if __name__ == "__main__":
    app.run_server()
