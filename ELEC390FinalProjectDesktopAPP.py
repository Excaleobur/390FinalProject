import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy import stats # imports from here and under do not appear to be used
from joblib import load
import h5py
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from tkinter import *

# UI and File Opening Procedure from https://realpython.com/python-gui-tkinter/

def load_model(file_path): # for loading trained models from separate training code
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def open_file():
    lbckg.grid_forget()
    
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("CSV", "*.csv"), ("All Files", "*.*")]
    )
    if not filepath:    # return and don't do anything if user closes window or cancels
        return
    txt_edit.delete("1.0", tk.END)  # clear existing text in window

    # Open the file with the filepath
    inputDataPD = pd.read_csv(filepath)

    # Rename columns to shorter names
    inputDataPD = inputDataPD.rename(columns={"Time (s)": "time", "Linear Acceleration x (m/s^2)": "x",
                                "Linear Acceleration y (m/s^2)": "y", "Linear Acceleration z (m/s^2)": "z",
                                "Absolute acceleration (m/s^2)": "abs"})

    window_size = 5 * 100  # Choose an appropriate window size (iphone 11 has 100hz rate so 5 * 100)

    # Calculate the moving average for each axis
    filtered_xi = inputDataPD['x'].rolling(window=window_size, center=True).mean()
    filtered_yi = inputDataPD['y'].rolling(window=window_size, center=True).mean()
    filtered_zi = inputDataPD['z'].rolling(window=window_size, center=True).mean()
    filtered_absi = inputDataPD['abs'].rolling(window=window_size, center=True).mean()

    # Combine filtered acceleration data with time into a new dataframe
    filtered_data_input = pd.DataFrame(
        {'time': inputDataPD['time'], 'x': filtered_xi, 'y': filtered_yi, 'z': filtered_zi, 'abs': filtered_absi}) #removed 'method': inputDataPD['method']
    filtered_data_input.dropna(inplace=True)  # Remove rows with NaN values due to the moving average calculation

    featuredInput = pd.DataFrame(columns=['max', 'min', 'mean', 'median', 'range', 'std', 'var', 'kurt', 'skew'])

    df_abs = filtered_data_input.iloc[:, 4]
    max = df_abs.rolling(window=window_size).max()
    min = df_abs.rolling(window=window_size).min()
    mean = df_abs.rolling(window=window_size).mean()
    median = df_abs.rolling(window=window_size).median()
    range = df_abs.rolling(window=window_size).apply(lambda x: x.max() - x.min())
    std = df_abs.rolling(window=window_size).std()
    var = df_abs.rolling(window=window_size).var()
    kurt = df_abs.rolling(window=window_size).kurt()
    skew = df_abs.rolling(window=window_size).skew()

    featuredInput['max'] = max
    featuredInput['min'] = min
    featuredInput['mean'] = mean
    featuredInput['median'] = median
    featuredInput['range'] = range
    featuredInput['std'] = std
    featuredInput['var'] = var
    featuredInput['kurt'] = kurt
    featuredInput['skew'] = skew

    featuredInput.dropna(inplace=True)
    featuredInput.fillna(0.0, inplace=True)
    #featuredInput.to_csv('inputFeatures.csv', index=False) # uncomment to output this incremental csv

    columns_to_normalize = ['max','min','mean','median','range','std','var','kurt','skew']
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(featuredInput[columns_to_normalize])
    featuredInput[columns_to_normalize] = normalized_data

    # Load the trained model
    clf = load_model('model.pkl')

    # The trained clf algorithm predicts the labels of the test input data based on its features. The predicted labels are stored in y_pred.
    y_pred = clf.predict(featuredInput)

    y_pred_series = pd.Series(y_pred, name="Prediction")
    y_pred_series.to_csv('y_predoutput.csv', index=False, header=True)

    # Print the number of rows in inputDataPD
    #print(f"Number of rows in inputDataPD: {inputDataPD.shape[0]}")
    # Print the number of entries in y_pred
    #print(f"Number of entries in y_pred: {len(y_pred)}")

    # Create a new column in inputDataPD named "Action_numerical" and "Action" with NaN values
    inputDataPD["Action_numerical"] = np.nan
    inputDataPD["Action"] = np.nan

    # Assign y_pred values to the new columns starting from the 499th row (Python uses 0-based indexing, so use 498)
    inputDataPD.loc[997:997 + len(y_pred) - 1, "Action"] = y_pred
    inputDataPD.loc[997:997 + len(y_pred) - 1, "Action_numerical"] = y_pred

    # Replace 1 with "walking" and 0 with "jumping" in the "Action" column
    inputDataPD["Action"] = inputDataPD["Action"].apply(lambda x: "walking" if x == 1 else ("jumping" if x == 0 else np.nan))

    #remove the Nan values from the dataframe
    inputDataPD = inputDataPD.dropna()

    # final csv output to current directory
    inputDataPD.to_csv('finalOutputWithLabels.csv', index=False)

    # change pd options so that prints or displays of data show the full thing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # put the inputDataPD into the text section of the window
    txt_edit.insert(tk.END, inputDataPD)

    # Extract time values from the filtered_data_input DataFrame
    time_values = filtered_data_input['time'].iloc[499:].values

    # Plot y_pred over time
    plt.plot(time_values, y_pred, marker='o', linestyle='')

    # Add labels and a title
    plt.xlabel('Time')
    plt.ylabel('Walking (1) or Jumping (0)')
    plt.title('Predicted Labels over Time')

    # Display the first plot
    plt.show()

    # Creates a figure with the number of subplots as there are columns in the csv arranged vertically
    fig, axs = plt.subplots(nrows=(len(inputDataPD.columns))-1, figsize=(10, 10))

    # Plot each column on a separate subplot
    for i, col in enumerate(inputDataPD.columns):
        if col == "Action": # don't plot the column with strings.
            continue
        axs[i].plot(inputDataPD[col])
        axs[i].set_title("Input Data " + col)
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(col)

    # Display the second window of plots
    fig.tight_layout()
    plt.show()

    window.title(f"ELEC 390 Group 35 Walk or Jump Identifier - {filepath}")

def save_file(): #Save button not used at the moment
    """Save the current file as a new file. This is already done with the open button but this allows more flexibility"""
    filepath = asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, mode="w", encoding="utf-8") as output_file:
        text = txt_edit.get("1.0", tk.END)
        output_file.write(text)
    window.title(f"ELEC 390 Group 35 Walk or Jump Identifier - {filepath}")

# Configure window and buttons
window = tk.Tk()
window.title("ELEC 390 Group 35 Walk or Jump Identifier")

#creating the overall window size and setting the background on the desktop application
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)
bckg = tk.PhotoImage(file = 'BackgroundGUI.png')
lbckg = Label(window, image=bckg)
lbckg.grid(row = 0, column = 0)

#creating buttons to use on the desktop

txt_edit = tk.Text(window)
frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_open = tk.Button(frm_buttons, text="Open", command=open_file, bg='#B8E2F2', activebackground='#808080')
btn_save = tk.Button(frm_buttons, text="Save As...", bg='#FF7E82', activebackground='#808080', command=save_file)

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=1, column=0, sticky="ew", padx=5)

frm_buttons.grid(row=0, column=0, sticky="ns")
txt_edit.grid(row=0, column=1, sticky="nsew")

window.mainloop()