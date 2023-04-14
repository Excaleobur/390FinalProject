import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from sklearn.metrics import f1_score

ds= pd.read_csv("C:/Users/Cassie G/Documents/390FinalProject-1/datasetOverall.csv") #add in name of dataset with path

#determine what column will be the data and which will be the labels
data = ds.iloc[:, 1:4]
labels = ds.iloc[:, 0]
                 
#determine how to divide each signal into 5-second windows


#assigning 10% of the data to the test data set. 
X_train, X_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)