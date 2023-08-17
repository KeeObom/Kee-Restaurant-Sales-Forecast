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

# # Model Training section has been put in model_build.py
# # Define parameters
# param_grid = {"n_estimators": [200,250,300], "min_samples_leaf": np.arange(1, 4), "max_features": [0.3, 0.4, 0.5,'sqrt'],"max_samples": np.arange(0.4, 0.7, 0.1)}

# # Working with the parameters
# grid2 = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=3)
# model = grid2.fit(X_train,y_train)
# print(model.best_params_,'\n')
# print(model.best_estimator_,'\n')

# # So after Tuning ang getting the best parameters for the RandomForest, let's use it and fit the the model
# rfcv = RandomForestRegressor(max_features=0.5, max_samples=0.6, n_estimators=200)
# rfcv.fit(X_train, y_train)

# Load trained model instead of training model each time program runs
from sklearn import model_selection, datasets
import joblib
import pickle

# Save the model using joblib
#joblib.dump(rfcv, "random_forest_model.joblib")

# Load the saved model
loaded_model = joblib.load("random_forest_model.joblib")
# # Load the model
# # Load model and use for predict
# loaded_model = joblib.load("rfcv_model_22.6.joblib")

# Make prediction
yhat = loaded_model.predict(X_test)
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
print('How well model explains trained data: ', loaded_model.score(X_train, y_train))

print('The R squared value is: ', r2_score(y_test, yhat))

print("RMSE", np.sqrt(mean_squared_error(y_test, yhat)))


# Get the weather function
def switch_weather(weather):
    if weather == 0:
        return "Clear Sky"
    elif weather == 1:
        return "Mainly Clear"
    elif weather == 2:
        return "Partly Cloudy"
    elif weather == 3:
        return "Overcast"
    elif weather == 45:
        return "Fog"
    elif weather == 48:
        return "Rime Fog"
    elif weather == 51:
        return "Light Drizzle"
    elif weather == 53:
        return "Moderate Drizzle"
    elif weather == 55:
        return "Dense Drizzle"
    elif weather == 56:
        return "Light Freezing Drizzle"
    elif weather == 57:
        return "Dense Freezing Drizzle"
    elif weather == 61:
        return "Slight Rain"
    elif weather == 63:
        return "Moderate Rain"
    elif weather == 65:
        return "Heavy Rain"
    elif weather == 66:
        return "Light Freezing Rain"
    elif weather == 67:
        return "Heavy Freezing Rain"
    elif weather == 71:
        return "Slight Snowfall"
    elif weather == 73:
        return "Moderate Snowfall"
    elif weather == 75:
        return "Heavy Snowfall"
    elif weather == 80:
        return "Slight Rain Showers"
    elif weather == 81:
        return "Moderate Rain Showers"
    elif weather == 82:
        return "Violent Rain Showers"
    elif weather == 85:
        return "Slight Snow Showers"
    elif weather == 86:
        return "Heavy Snow Showers"
    elif weather == 95:
        return "Thunderstorm"
    elif weather == 96:
        return "Thunderstorm with Slight Hail"
    elif weather == 99:
        return "Thunderstorm with Heavy Hail"
#    THE DASH APP
import dash_bootstrap_components as dbc

# make cards function
weather1 = '/assets/weather2.jpg'
n = 0
def make_card(title, amount, weather, status):
    global weather1
    if status == 3:
        weather1 = '/assets/sunny.JPG'
    elif status == 0:
        weather = '/assets/sunny.JPG'
    elif status == 1:
        weather1 = '/assets/sunny.JPG'
    elif status == 80:
        weather1 = '/assets/rain.JPG'
    elif status == 81:
        weather1 = '/assets/rain.JPG'
    elif status == 82:
        weather1 = '/assets/rain.JPG'
    elif status == 85:
        weather1 = '/assets/rain.JPG'
    elif status == 53:
        weather1 = '/assets/two_cloud.JPG'
    elif status == 55:
        weather1 = '/assets/two_cloud.JPG'
    elif status == 51:
        weather1 = '/assets/two_cloud.JPG'
    elif status == 45:
        weather1 = '/assets/fog.JPG'
    elif status == 61:
        weather1 = '/assets/cloud.JPG'
    elif status == 63:
        weather1 = '/assets/cloud.JPG'
    elif status == 65:
        weather1 = '/assets/cloud.JPG'
    elif status == 2:
        weather1 = '/assets/cloud.JPG'
    elif status == 71:
        weather1 = '/assets/fog.JPG'
    elif status == 73:
        weather1 = '/assets/fog.JPG'
    elif status == 75:
        weather1 = '/assets/fog.JPG'
    elif status == 95:
        weather1 = '/assets/thunderstorm.JPG'
    elif status == 96:
        weather1 = '/assets/thunderstorm.JPG'
    elif status == 99:
        weather1 = '/assets/thunderstorm.JPG'

    return dbc.Card(
        [
            dbc.CardHeader(html.H2(f"₦{title}"), style={"background": "red", "maxWidth": 350, 'height': '50px',
                   'border': '0px',
                   'borderRadius': '5px', 'backgroundColor':
                   'black', 'color': 'white', 'textTransform':
                   'uppercase', 'fontSize': '9px'}),
            # src='/assets/weather2.jpg'
            dbc.CardImg(src=weather1, top=True, style={"maxWidth": 350, "maxheight": 175}),
            dbc.CardBody(html.H5(f"{amount} {weather}", id=title), style={"color": "black", "background": "LightCoral", "maxWidth": 350, "maxheight": 70, 'fontSize': '20px', 'border': '0px','borderRadius': '5px'}),
        ],
         style={"width": "13rem"}, # className="text-center shadow", style={"maxwidth": 175},
    )


# Actual vs predicted figure
fig = px.line(df_rf, x=df_rf.index, y=["Actual", "Predicted"], title="Actual vs Predicted Sales", template="seaborn")

# fig = px.line(df_rf, x=df_rf.index, y=["Actual", "Predicted"])
# return fig
import datetime
begin = date.today()


app = Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL]) # JOURNAL or
server = app.server


summary = pd.DataFrame({"₦₦₦": ["₦₦₦_","₦|₦₦","₦₦|₦"], "Weather_": ["Weather","Weather","Weather" ], "Pic": ["Pic","Pic","Pic"], "Status": ["Status", "Status", "Status"]})

app.layout = html.Div([

    html.H1(id="Title", children="RESTAURANT DAILY SALES FORECAST", style={'textAlign': 'center'}),

    html.H2(id="prediction_result", children="Pick Date Range: "),

    # dcc.DatePickerSingle(
    #     id='my-date-picker-single',
    #     min_date_allowed=date(1995, 8, 5),
    #     max_date_allowed=date(2030, 12, 31),
    #     initial_visible_month=date(2023, 1, 1),
    #     date=date.today()
    # ),
    # html.Div(id='output-container-date-picker-single'),

    dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date(2030, 12, 31),
        initial_visible_month=date.today(),#date(2023, 1, 1),
        #end_date=date.today(),
        start_date=date.today(),
        #end_date=begin + datetime.timedelta(days=7)
    ),
    html.Div(id='output-container-date-picker-range'),
    html.Br(),

    dcc.Graph(id='line_plot', figure={}),
    html.Br(),

    html.Br(),
    html.H2(id="table-header", children="FORECAST TABLE", style={'textAlign': 'center'}),
    html.Table(id="table-container", children={}, style={'marginLeft': 'auto', 'marginRight': 'auto'}),

    html.Br(),

    # CARDS CARDS CARDS
    html.H2(id="cards-header", children="INFO CARDS", style={'textAlign': 'center'}),
    dbc.Container(
    #html.Div(
        dbc.Row([dbc.Col(make_card(k, v, weather_stat, status)) for k, v , weather_stat, status in summary.itertuples(index=False)],  style={"width":"800px"}), # The 800px sets the behind border size
         className="p-5", id="card_container",
    ),

    html.A("@keeobom🐦", href="https://twitter.com/keeobom"),
    

])



@app.callback([# Output(component_id="prediction_result", component_property="children"),
                Output(component_id='line_plot', component_property='figure'),
               Output(component_id="table-container", component_property="children"),
               Output(component_id="card_container", component_property="children"),
               ],

              [Input(component_id="my-date-picker-range", component_property="start_date"),
               Input(component_id='my-date-picker-range', component_property='end_date'),
               
               ])
def make_prediction(start_date, end_date):
    if start_date is not None:

        # Set start and end date for API call
        # Have to do pick a date or pick a range, as past days can't work with past7days
        # The format of dates is 2023-01-10
        start_date_1 = start_date #"2023-01-17"
        end_date_1 = end_date #"2023-01-22"
        oneday_api = "https://api.open-meteo.com/v1/forecast?latitude=4.78&longitude=7.01&hourly=temperature_2m,windspeed_10m,relativehumidity_2m,pressure_msl&windspeed_unit=mph&timezone=Africa/Lagos&past_days=7&daily=weathercode"
        period_api = f"https://api.open-meteo.com/v1/forecast?latitude=4.78&longitude=7.01&hourly=temperature_2m,windspeed_10m,relativehumidity_2m,pressure_msl&windspeed_unit=mph&timezone=Africa/Lagos&daily=weathercode&start_date={start_date_1}&end_date={end_date_1}"
        # You can either choose today and it shows past and future or you pick a range

        # Make weather API call
        import requests
        weather_api = period_api
        r = requests.get(weather_api)
        json = r.json()

        # Get needed parameters as list
        time_list = json['hourly']['time']
        temp_list = json['hourly']['temperature_2m']
        wind_list = json['hourly']['windspeed_10m']
        humd_list = json['hourly']['relativehumidity_2m']
        pres_list = json['hourly']['pressure_msl']
        weather_code = json['daily']['weathercode']

        # Convert the dates
        from dateutil import parser
        time_list_1 = []
        for i in range(len(time_list)):
            test_time = parser.parse(time_list[i])
            test_time = test_time.strftime("%Y-%m-%d")
            time_list_1.append(test_time)
        # print(time_list_1, len(time_list_1))

        # Convert hpa to inHg
        pres_list_1 = []
        for i in range(len(pres_list)):
            test_pres = round((pres_list[i] * 0.02952998330101), 2)
            pres_list_1.append(test_pres)
        # print(pres_list_1, len(pres_list_1))

        # Make dataframe of hourly values
        hourly_df = pd.DataFrame(
            {"Date": time_list_1, "Temp": temp_list, "Wind": wind_list, "Humidity": humd_list, "Pressure": pres_list_1})
        # hourly_df

        time_set = set(time_list_1)
        time_set = list(time_set)

        #### Remove from 12am/24:00 to 5am, and remove 10pm-12pm rows, as weather here does not affect genesis sales
        # want to remove first
        hourly_test = hourly_df.copy()
        df_list = []
        for i in time_set:
            # df = df[df['Credit-Rating'].str.contains('Fair')]
            df_date = hourly_test[hourly_test['Date'].str.contains(i)]
            # grouped = hourly_test.groupby(hourly_test.Date)
            # df1 = grouped.get_group("2023-01-09")
            df_date.drop(df_date.head(6).index, inplace=True)  # drop first n rows
            df_date.drop(df_date.tail(1).index, inplace=True)  # drop last n rows
            df_date = df_date.reset_index()
            df_date = df_date.drop('index', axis=1)
            df_list.append(df_date)
        # df_list[0]

        # Now to add the list of dataframes into one dataframe
        df_info = pd.concat(df_list)
        df_info = df_info.reset_index()
        df_info = df_info.drop('index', axis=1)
        # df_info

        # Group Date column and return the mean or average of the other columns
        hourly_grouped = df_info.groupby('Date').mean()
        # hourly_grouped

        # Now we have dataframe containing past and future 7 days
        hourly_grouped = hourly_grouped.reset_index()

        # Expand Date
        groupedd = hourly_grouped.copy()
        import datetime as dt
        groupedd['Date'] = pd.to_datetime(groupedd['Date'], format="%Y-%m-%d")

        # Expand the date
        # Now it has been converted to datetime, let's extract the month, week, day of month and day of week from the date
        groupedd['Date_month'] = groupedd['Date'].dt.month
        groupedd['Date_day'] = groupedd['Date'].dt.day
        groupedd['Date_dayofweek'] = groupedd['Date'].dt.dayofweek
        # groupedd

        #### So the dash datepicker returns date in the format "YYYY-MM-DD" as a string
        from datetime import date



        try:
            # Return PREDICTIONS for past 7 and future 7 days
            value_list = []
            day_list = []
            for i in groupedd["Date"]:
                # date_value = "2023-01-17"
                # date_object = date.fromisoformat(i)

                df_pred = groupedd[groupedd['Date'] == i]
                df_prediction1 = pd.DataFrame(
                    {'Tavg': df_pred['Temp'], 'Havg': df_pred['Humidity'], 'Wavg': df_pred['Wind'],
                     'Pavg': df_pred['Pressure'], 'Date_day': df_pred['Date_day'],
                     'Date_dayofweek': df_pred['Date_dayofweek']})
                X_prediction1 = df_prediction1[
                    ['Tavg', 'Havg', 'Wavg', 'Pavg', 'Date_day', 'Date_dayofweek']].to_numpy()

                # Use standard scaler
                input_X = sc.transform(X_prediction1)
                # make prediction with the model rfcv
                rf_prediction1 = loaded_model.predict(input_X)
                print(f'{round(rf_prediction1[0]):,}', i.strftime('%A'))

                # convert results to a dataframe
                value_list.append(round(rf_prediction1[0]))
                day_list.append(i.strftime('%A %d %b'))
            fin_df = pd.DataFrame({'Sales': value_list, 'Days': day_list})

            # Plot the lineplot
            fig1 = px.line(fin_df, x='Days', y='Sales', markers=True)

            fig1.update_layout(
                plot_bgcolor='#FDEDEC',
                paper_bgcolor='#F2D7D5',
                font_color='#212F3C'
            )

            # container = "Predicted Sales: N{}".format(fin_df)

            # work on table before display
            # swap column positions and show thousandth values
            fin_df1 = fin_df.copy()
            columns_titles = ["Days", "Sales"]
            fin_df1 = fin_df1.reindex(columns=columns_titles)

            # Get thousandth values
            for sale in fin_df1['Sales']:
                fin_df1['Sales'].replace(sale, f'{sale:,}', inplace=True)
            fin_df1.rename(columns={"Sales": "Sales (₦)"}, inplace=True)

            # Get the dataframe to a table
            import dash_bootstrap_components as dbc
            table = dbc.Table.from_dataframe(fin_df1, striped=True, bordered=True, hover=True)

            # Add weather code to dataframe
            fin_df2 = fin_df1.copy()
            fin_df2['weather'] = [switch_weather(i) for i in weather_code]
            fin_df2['status'] = weather_code

            # change weather picture for each loop
            print(fin_df2['status'])



            # cards
            cardd = dbc.Container(
                # html.Div(
                dbc.Row([dbc.Col(make_card(v, k, weather_stat, status)) for k, v, weather_stat, status in fin_df2.itertuples(index=False)], style={"width": "1000px"}),
                # The 800px sets the behind border size
                className="p-5", id="card_container",
            ),


            return fig1, table, cardd #, container


        except ValueError:
            return "Please indicate the date range"


if __name__ == '__main__':
    app.run_server(debug=True)
