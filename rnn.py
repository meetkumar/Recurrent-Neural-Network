#Recurrent Neural Networks

#Part 1 - Data Preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Inserting the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values #Only considering Open stock price of the stock

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) #Normalization feature scaling x-min(x) / max(x) - min(x)

#Creating a data structure with 60 timesteps and 1 output
#At each time t, RNN is going to look 60 days before time t in order to predict output based on the patterns for time t+1
X_train = []
y_train = []
for i in range(60, 1258):
	X_train.append(training_set_scaled[i-60:i, 0])
	y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))