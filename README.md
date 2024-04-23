# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
   

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: santhosh kumar B
RegisterNumber: 212223230193 
*/
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

## Output:

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/2c96e209-e2b3-4c0f-8eea-c8e05e8420e2)

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/0e97fcd6-968b-46ef-9be4-03f1ce1e8679)

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/95b01329-6f39-4a7a-b6e2-8fb683575320)

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/fe84021e-d8d2-44e9-a0e9-876466727cd9)

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/84b9a555-328e-49a6-b1c6-a59c44ae0e51)

![image](https://github.com/Santhoshstudent/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446853/2babbfb3-1dde-40ba-882a-41bee5668774)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
