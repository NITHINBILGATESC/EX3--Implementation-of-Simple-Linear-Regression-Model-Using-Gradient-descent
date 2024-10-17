# EX3 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.We will use basic libraries like numpy for numerical computations and matplotlib for plotting graphs. 
2.let's generate a synthetic dataset for this example. Assume the relationship between the independent variable X and the dependent variable y is linear.
3.Initialize parameters
4.For Linear Regression, we use Mean Squared Error (MSE) as the cost function:

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NITHIN BILGATES C
RegisterNumber: 2305001022 
*/
```import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):

  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv('/content/50_Startups (1).csv',header=None)
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")
```
## Output:
![image](https://github.com/user-attachments/assets/9cf62ed8-380d-46d3-9981-25e647ccdc12)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
