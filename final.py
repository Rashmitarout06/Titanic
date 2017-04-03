# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:22:23 2016

@author: Rashmita Rout
"""

#importing pandas and numpy
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#first we must begin with importing the train and test dataset
train = pd.read_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/train.csv")
test = pd.read_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/test.csv")

#converting the test and train datasets into dataframes
train_df= pd.DataFrame(train)
test_df= pd.DataFrame(test)

#we check both the datasets to find the columns present in them and the find the number of missing values in each column
train_df.info()
test_df.info()

#We find that the test dataset does not have the "Survived" column.
#To combine both the dataset we need to add the "survived" column to the test dataset.
test_df['Survived']= [None]*418

#To perform imputations we need to combine both the train and test datasets
frames = [train_df,test_df]
data_total = pd.concat(frames)

#To analyze the combined dataset we use the info function
data_total.info()

#we see that in the combined dataset "Age", "Cabin", "Emnarked" and "Fare" columns have missing values.
#Also we see that PassengerId ,Parch and SibSp columns can be combined into a single column
#hence combining the PassengerId ,Parch and SibSp into one column as "fam_size"
data_total['fam_size'] = data_total['Parch'] + data_total['SibSp'] +1

# Our next step is to fill all the missing values in each column efficiently.

#Fare column has only one missing value, from our dataset we see that the passenger is of Pclass = 3, 
#hence filling the missing values in the Fare column with the mean of Pclass = 3 which is 12.45967788
data_total.iat[1043,3] = 12.45967788

#Embarked column has two missing values,
#filling the missing values in the Embarked column with the most frequent value
data_total['Embarked'].fillna('S', inplace=True)

#Age column has many missing values 
#We predict the missing age values by extracting the titles from the "Name" column 
#and categorizing them into "Mr", "Miss", "Master", "Mrs" and "Others" and changing them into numerical values
import re
for i in range(0,1308):
    data_total.iat[i,4] = re.split('[.,]',data_total.iat[i,4])[1]
    
data_total['Name'] = data_total['Name'].map({' Capt':4,' Col':4,' Don':4,\
' Dona':4, ' Dr':4, ' Jonkheer':4, ' Lady':1,\
       ' Major':4, ' Master':2, ' Miss':3, ' Mlle':3, ' Mme':3, ' Mr':0,\
       ' Mrs':1, ' Ms':3,\
       ' Rev':4, ' Sir':4, ' the Countess':4, 'Peter, Master. Michael J':4})

#writing a function to find the mean age by comparing the "Name", "Pclass" and "Sex"
avgAgeByTitle = data_total.groupby(['Name','Pclass','Sex']).agg({'Age': np.median})
def returnAvgAge(title,cls,gender):
    return avgAgeByTitle.ix[title].ix[cls].ix[gender]['Age']

for i in range(0,1308):
    if(data_total['Age'].isnull().values[i]):
        data_total.iat[i,0] = returnAvgAge(data_total.iat[i,4],data_total.iat[i,7],data_total.iat[i,8])

data_total.iat[1308,0] = 6.00000        
    
#dropping the variables that are making the model weak and which are statistically not much significant variables 
data_total=data_total.drop(['Ticket','Cabin','Parch','SibSp','PassengerId','Name','Age'], axis=1)

#changing the categorical values into numerical values
data_total['Sex'] = data_total['Sex'].map({'male':1,'female':0})
data_total['Embarked'] = data_total['Embarked'].map({'S':0,'C':1,'Q':2})
data_total['fam_size'] = data_total['fam_size'].map({1:1,2:2,3:2,4:2,5:3,6:3,7:3,8:3,11:3})

#Now that we have done all the imputations we will create the features and labels
#creating the train and test features and labels
train_labels = train.Survived
train_features = data_total.ix[:890]
train_features = train_features.drop(['Survived'],axis=1)
test_features = data_total.iloc[891:1309,:]
test_features = test_features.drop(['Survived'],axis=1)

##using random forest classifier 
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=1000)
clffit = clf.fit(train_features,train_labels)
pred= clf.predict(test_features)
print(clffit.score(train_features,train_labels))

#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 
final= pd.DataFrame()
final['PassengerId']= test.PassengerId
final['Survived']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/final_1.csv",index=False)



###using Extra tree classifier
from sklearn import tree
etcclf= tree.ExtraTreeClassifier(criterion='gini', splitter='random')
etcfit = etcclf.fit(train_features,train_labels)
pred= etcclf.predict(test_features)
print(etcfit.score(train_features,train_labels))

#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 
final= pd.DataFrame()
final['PassengerId']= test.PassengerId
final['Survived']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/final_2.csv",index=False)



##using decision tree classifier
from sklearn import tree
dtcclf = tree.DecisionTreeClassifier()
dtcfit = dtcclf.fit(train_features,train_labels)
pred= dtcclf.predict(test_features)
print(dtcfit.score(train_features,train_labels))

#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 
final= pd.DataFrame()
final['PassengerId']= test.PassengerId
final['Survived']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/final_3.csv",index=False)




##using gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
gbmfit = gbm.fit(train_features,train_labels)
pred = gbm.predict(test_features)
print(gbmfit.score(train_features,train_labels))

#importing the predicted values into excel file in two columns "PassengerId" and "Survived" 
final= pd.DataFrame()
final['PassengerId']= test.PassengerId
final['Survived']= pred

#submitting the predicted values in a csv file
final.to_csv("C:/Users/Rashmita Rout/Desktop/Titanic Data/final/final_4.csv",index=False)


