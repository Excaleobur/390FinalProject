import numpy as np
import pandas as pd
from scipy import stats
from joblib import load
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter.font as font
from tkinter import ttk


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def process_data(input_file_path):

    # Read input data file
    input_data = pd.read_csv(input_file_path)

    #renaming the columns in the dataset so it is easier to call - column names (time, x,y,z,abs) are the same for all datasets
    input_data = input_data.rename(columns={"Time (s)" : "time", "Linear Acceleration x (m/s^2)" : "x", 
                        "Linear Acceleration y (m/s^2)" : "y", "Linear Acceleration z (m/s^2)" : "z", 
                        "Absolute acceleration (m/s^2)" : "abs"})
    
    window_size = 5 * 100  #Choose an appropriate window size (iphone 11 has 100hz rate so 5 * 100)

    # Calculate the moving average for each axis
    filtered_xj = input_data['x'].rolling(window=window_size, center=True).mean()
    filtered_yj = input_data['y'].rolling(window=window_size, center=True).mean()
    filtered_zj = input_data['z'].rolling(window=window_size, center=True).mean()
    filtered_absj = input_data['abs'].rolling(window=window_size, center=True).mean()

     # Combine filtered acceleration data with time into a new dataframe
    filtered_data_jump = pd.DataFrame({'time': input_data['time'], 'x': filtered_xj, 'y': filtered_yj, 'z': filtered_zj, 'abs': filtered_absj})
    filtered_data_jump.dropna(inplace=True)  # Remove rows with NaN values due to the moving average calculation
    
    featured = pd.DataFrame(columns = ['max', 'min', 'mean', 'median', 'range', 'std', 'var', 'kurt', 'skew'])

    df_abs = filtered_data_jump.iloc[:,4]
    max = df_abs.rolling(window= window_size).max()                 
    min = df_abs.rolling(window= window_size).min()
    mean = df_abs.rolling(window= window_size).mean()
    median =df_abs.rolling(window= window_size).median()
    range = df_abs.rolling(window= window_size).apply(lambda x: x.max() - x.min())
    std = df_abs.rolling(window= window_size).std()
    var = df_abs.rolling(window= window_size).var()
    kurt = df_abs.rolling(window= window_size).kurt()
    skew = df_abs.rolling(window= window_size).skew()

    featured['max'] = max                
    featured['min'] = min
    featured['mean'] = mean
    featured['median'] = median
    featured['range'] = range
    featured['std'] = std
    featured['var'] = var
    featured['kurt']= kurt
    featured['skew'] = skew

    featured.dropna(inplace=True)

    result = featured
    result.fillna(0.0, inplace=True)

    #normalize the data
    columns_to_normalize = ['max','min','mean','median','range','std','var','kurt','skew']
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(result[columns_to_normalize])
    result[columns_to_normalize] = normalized_data

    return result


def predict(model, input_data):
    return model.predict(input_data)


def browse_button():
    file_path = filedialog.askopenfilename()
    input_data = process_data(file_path)
    prediction = predict(clf, input_data)
    result_label.config(text=f'Prediction: {prediction}')


# Load the trained model
clf = load_model('model.pkl')


# Create the GUI
root = tk.Tk()
root.title("Walking or Jumping Predictor")
root.geometry("400x300")  # Adjust the size of the window

frame = ttk.Frame(root, padding="20")
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Adjust the button size and font style
button_font = font.Font(size=14, weight="bold")
style = ttk.Style()
style.configure("TButton", font=button_font)

button = ttk.Button(frame, text="Browse", command=browse_button, style="TButton")
button.grid(row=0, column=0, pady=(0, 20))  # Add padding to separate the Browse button from the Prediction text

result_font = font.Font(size=12, weight="bold")
result_label = ttk.Label(frame, text="Prediction: ", font=result_font)
result_label.grid(row=1, column=0)

root.mainloop()




