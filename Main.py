# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:53:53 2022

@author: rahul
"""

#Packages to be imported 
import pandas as pd 
import numpy as np 

#Importing the train and test data as a DataFrame
Train_df = pd.read_csv("C:/Users/rahul/Desktop/hackNintn/Datahack/train_qWM28Yl.csv")
Test_df =  pd.read_csv("C:/Users/rahul/Desktop/hackNintn/Datahack/test_zo1G9sv.csv")

#DataPreprocessing 

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
attributes = list(Train_df.columns)
attributes.remove('policy_id')
attributes.remove('is_claim')

# Encode labels in column 'species'.
for att in attributes:
    Train_df[att]= label_encoder.fit_transform(Train_df[att])
    Test_df[att]= label_encoder.fit_transform(Test_df[att])


# Train_df.to_csv(r"C:\Users\rahul\Desktop\hackNintn\Datahack\train_edt1.csv")
# Test_df.to_csv(r"C:\Users\rahul\Desktop\hackNintn\Datahack\test_edt1.csv")


input_train = Train_df[attributes].values
output_train = Train_df['is_claim'].values
# input_test = Test_df[attributes].values
# insu_ID = Test_df['policy_id'].values

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y =train_test_split(input_train,output_train,test_size=0.25,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_x,train_y)
ydash = classifier.predict(test_x)
print(ydash)
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
cm = confusion_matrix(test_y,ydash)
asc = accuracy_score(test_y,ydash)
#print("Confusion Matrix: \n",cm)
#print("Accuracy_score: ",asc*100)
print('F1_score',f1_score(test_y, ydash, average='weighted'))


