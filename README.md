# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries & load data using pandas, and preview with df.head().

2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.

3.Encode categorical columns (like gender, education streams) using LabelEncoder.

4.Split features and target:

X = all columns except status

y = status (Placed/Not Placed)

5.Train-test split (80/20) and initialize LogisticRegression.

6.Fit the model and make predictions.

7.Evaluate model with accuracy, confusion matrix, and classification report.

## Program :
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Elavarasan M
RegisterNumber:  212224040083
*/
```

```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
```
```
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
```
data1.isnull().sum()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```


```
x=data1.iloc[:, : -1]
x
```

```
y=data1["status"]
y
```


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
```
```
print(model.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]))
```

## Output:

**Head Values**

![Screenshot 2025-04-27 121531](https://github.com/user-attachments/assets/c0794227-035a-4b15-adb4-d91a63500c0d)

**Head Values - Copy(Dropped)**

![Screenshot 2025-04-27 122555](https://github.com/user-attachments/assets/f993e35d-969a-4df2-bd66-abbe41a33ea9)

**Sum - Null Values**

![Screenshot 2025-04-27 122704](https://github.com/user-attachments/assets/e9cb6c73-770e-40b5-8861-fd1c833d971b)

**Sum - Duplicated**

![Screenshot 2025-04-27 122825](https://github.com/user-attachments/assets/64221f25-741b-41ac-b5db-1c3db7ef1e3a)

**LabelEncoded**

![Screenshot 2025-04-27 122926](https://github.com/user-attachments/assets/4ee4d111-419b-46a7-b062-59a939d9c264)

**X values**

![Screenshot 2025-04-27 123425](https://github.com/user-attachments/assets/1f2ff4f0-7260-4c7d-828b-52a00f8f306d)

**Predicted Y values**

![Screenshot 2025-04-27 123445](https://github.com/user-attachments/assets/4eef24d5-5601-442f-80c5-7b2df92b62b2)

**Accuracy Score,Confusion matrix and Classfication report**

![Screenshot 2025-04-27 123518](https://github.com/user-attachments/assets/5a1e4b65-3ec0-4069-bcbb-1a19f1eb94a1)

**Model prediction for input**


![Screenshot 2025-05-12 171604](https://github.com/user-attachments/assets/d6ae591d-bd0f-406c-9d41-be623ec50fcf)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
