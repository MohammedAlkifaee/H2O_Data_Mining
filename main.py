from operator import truediv
import requests
import MetaTrader5 as mt5
import pandas as pd
import time
import sqlite3
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import chardet

h2o.init()

# Connect to SQLite database
conn = sqlite3.connect("forex_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS forex_ (
    Date TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL
)
""")
conn.commit()

ascii_art = """
****************************************************************************************************
*  _   _ _   _ _____     _______ ____  ____ ___ _______   __   ___  _____   _  ___   _ _____ _     *
* | | | | \ | |_ _\ \   / / ____|  _ \/ ___|_ _|_   _\ \ / /  / _ \|  ___| | |/ / | | |  ___/ \    *
* | | | |  \| || | \ \ / /|  _| | |_) \___ \| |  | |  \ V /  | | | | |_    | ' /| | | | |_ / _ \   *
* | |_| | |\  || |  \ V / | |___|  _ < ___) | |  | |   | |   | |_| |  _|   | . \| |_| |  _/ ___ \  *
*  \___/|_| \_|___|  \_/  |_____|_| \_\____/___| |_|   |_|    \___/|_|     |_|\_\\___/|_|/_/   \_\ *
****************************************************************************************************
                         by Dr.Ahmed Al-Janaby  & Mohammad Abbas Shareef
"""

print(ascii_art)
input("Enter any key to continue ...")

cursor.execute("SELECT * FROM forex_")
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
hf = h2o.H2OFrame(df)

hf.head()

# Define target and features
target = 'Close'
features = ['Open', 'High', 'Low']

# Split data into training and testing sets
train, test = hf.split_frame(ratios=[0.8], seed=1234)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=target, training_frame=train)

lb = aml.leaderboard

# Get the best model and make predictions
best_model = aml.leader
predictions = best_model.predict(test)
print(predictions.head())

# Evaluate model performance
performance = best_model.model_performance(test)

print("RMSE: ", performance.rmse())
print("MAE: ", performance.mae())
print("MSE: ", performance.mse())
print("R^2: ", performance.r2())

train_performance = best_model.model_performance(train)

print("\nTrain Data Evaluation:")
print("RMSE: ", train_performance.rmse())
print("MAE: ", train_performance.mae())
print("MSE: ", train_performance.mse())
print("R^2: ", train_performance.r2())

print("\nMean Residual Deviance on Test Set: ", performance.mean_residual_deviance())
