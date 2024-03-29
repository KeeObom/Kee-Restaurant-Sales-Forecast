# Restaurant Sales Forecasting with Random Forest Regressor

This project builds a Random Forest Regressor model to forecast daily sales for a restaurant. The model uses weather conditions obtained from the open-meteo API to predict future sales. The UI is built using Dash Plotly, allowing users to select a date range for forecasting and visualize weather conditions.

## Features

- Random Forest Regressor model for daily sales forecasting.
- Dash Plotly UI for user interaction and visualization.
- Integration with the open-meteo(https://open-meteo.com/) API to fetch weather conditions.
- Easy retraining of the model using `model_build.py`.
- Pre-trained Random Forest model saved as `random_forest_model.joblib`.
- Past daily sales data available in `sales_science_csv.csv`.

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the `app.py` file to start the Dash Plotly web application.
3. Access the UI in your web browser at `http://localhost:8050`.
4. Choose a date range for forecasting, and the app will display sales predictions and weather conditions.

## Files

- `app.py`: Main file containing the Dash Plotly application.
- `model_build.py`: Script to rebuild the Random Forest model using past sales data.
- `random_forest_model.joblib`: Pre-trained Random Forest model.
- `sales_science_csv.csv`: Past daily sales data.
- `requirements.txt`: List of required packages for the project.

## Notes

- The open-meteo API is free.
- The `model_build.py` script can be used to retrain the Random Forest model based on new sales data.

## License

Feel free to contribute, report issues, or provide suggestions to improve the project.

---

Created by Barinatamkee Melu-Akekue
