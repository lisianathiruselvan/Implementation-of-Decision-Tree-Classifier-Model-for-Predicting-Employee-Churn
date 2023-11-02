# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: T.LISIANA
RegisterNumber: 212222240053
*/
import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## Data Head:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/f313b4fc-4610-4013-b21f-0342bf27a883)


## Data set info:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/6cda7463-c893-4822-9177-c682fd6a30f0)

## Null dataset:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/90e72038-146a-4a49-8cd3-52a1e1d423eb)


## Values count in left column:


![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/2c47d510-8b49-4649-afb6-9331068126f7)


## Dataset transformed head:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/9ae64c64-747b-46f4-b8fc-5cd9b0de1353)


## x.head:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/2e5a9e6c-8434-4f08-8182-e23c76388143)


## Accuracy:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/a3f5c109-3ca4-4399-a3c9-3bfa0390c183)


## Data Prediction:

![image](https://github.com/lisianathiruselvan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119389971/67091f5f-132d-4c13-8898-4ffb2f194f52)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
