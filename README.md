# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NIVESHA P
RegisterNumber:  212222040108

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")![s1](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/4525d7f6-1601-4746-bb1e-0096b6da4766)

plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![s1](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/61856f86-fcc8-4fae-a661-5e121316b79f)
![s2](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/be2a4c68-2c27-4f57-9bd8-3cc541907fcb)
![s3](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/a6ea7e9c-4d56-40a9-b667-7eb84de6c09c)
![s4](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/275ee52b-40d5-4aaa-89e9-1d4b78414c9a)
![s5](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/22ed5fe9-1388-4b92-bcac-e07174439136)
![s6](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/1018dbe7-51a9-401d-a3e9-c9ec111e8488)
![s7](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/1dfff050-bcd7-4f10-b98a-79937c8f2281)
![s8](https://github.com/niveshaprabu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122986499/28862ff7-b7ff-4bcf-b17f-6ee8071cc66d)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
