#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 07:12:13 2021

@author: buggzy
"""
#import libarries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#read.csv
dataset = pd.read_csv('BankCustomers.csv')
dataset.head()

#defining independent / dependent feature
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

#using dummies to expand catogorical fields

gender =pd.get_dummies(X['Gender'],drop_first=True)
state = pd.get_dummies(X['Geography'],drop_first=True)

#add above in X
X=pd.concat([X,state,gender],axis=1)

#drop usless coloumn
X=X.drop(['Gender','Geography'],axis=1)

#test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Deep Learning start
import keras
from keras.models import Sequential
from keras.layers import Dense

#Create hidden layers using dense

Classifier = Sequential()

Classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

Classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

Classifier.add(Dense(activation="relu", units=1, kernel_initializer="uniform"))


#Compiling ANN function
Classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting ann to the training set
Classifier.fit(X_train,y_train, batch_size = 10, epochs=50)

#predicting test dataset
y_pred = Classifier.predict(X_test)
y_pred = (y_pred > 0.5)





