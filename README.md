![image](https://github.com/user-attachments/assets/0600a5be-24d1-4be8-b4a2-8276ed2585b6)
# Forex Price Prediction using H2O AutoML

## Project Overview
This project focuses on predicting the closing prices in the Forex market using machine learning techniques. By leveraging H2O AutoML, the model aims to analyze historical data and make accurate forecasts to assist in financial decision-making.

## Authors
- **Mohammad Abbas Shareef**  
- **Dr. Ahmed Al-Janaby**

## Project Description
The project automates the process of training and selecting the best machine learning models to predict Forex prices. H2O AutoML is used to train multiple models and select the one with the highest accuracy. The data is processed and stored in an SQLite database, and real-time Forex data can be fetched directly from MetaTrader 5 (MT5).

## Features
- Fetch live Forex data using MetaTrader 5 API.
- Store and manage historical Forex data in an SQLite database.
- Apply H2O AutoML to train models and predict closing prices.
- Evaluate model performance using RMSE, MAE, MSE, and R-squared metrics.
- Visual ASCII art display upon program execution.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `pandas`
  - `h2o`
  - `sqlite3`
  - `MetaTrader5`
  - `sklearn`
  - `requests`
- **Database:** SQLite

## How to Run the Project
1. Install the required libraries:
   ```bash
   pip install pandas h2o MetaTrader5 chardet scikit-learn
   ```
2. Initialize H2O:
   ```python
   h2o.init()
   ```
3. Connect to SQLite and create the necessary table:
   ```python
   conn = sqlite3.connect("forex_data.db")
   cursor = conn.cursor()
   ```
4. (Optional) Import historical Forex data from CSV to SQLite:
   ```python
   df.to_sql("forex_", conn, if_exists="append", index=False)
   ```
5. Fetch real-time data from MetaTrader 5 (MT5):
   ```python
   get_live_data("EURUSD")
   ```
6. Train the model and evaluate its performance:
   ```python
   aml.train(x=features, y=target, training_frame=train)
   ```

## Model Performance
The model's performance is evaluated using:
- **RMSE (Root Mean Square Error):** Measures model accuracy.
- **MAE (Mean Absolute Error):** Measures average prediction error.
- **MSE (Mean Squared Error):** Average of squared differences.
- **R-squared (R^2):** Measures how well the model explains variance in data.

## Example Output
```
RMSE: 0.0156
MAE: 0.0102
R^2: 98.6%
```

## Notes
- Ensure that MetaTrader 5 is installed and configured on your system.
- The script can fetch live data continuously until interrupted (Ctrl + C).
- ASCII art displays upon program start for aesthetics.

## Disclaimer
This project is for educational purposes only and should not be considered financial advice.

