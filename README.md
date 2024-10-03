# EX3 Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Set up the initial values for the model parameters and define hyperparameters that control the learning process.
2. Use the current parameters to make predictions and evaluate the performance of the model using a cost function.
3. Determine the direction and magnitude by which to adjust the parameters to minimize the cost function.
4. Adjust the model parameters in the direction that reduces the cost and iterate the process until convergence or the maximum number of iterations is reached.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iterations=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    for _ in range(num_iterations):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)

        theta -= learning_rate * (1/len(X1))  * X.T.dot(errors)

    return theta

    from re import S
data = pd.read_csv('50_Startups.csv',header=None)
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:, -1].values).reshape(-1, 1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")

## Output:
![image](https://github.com/user-attachments/assets/614e24fe-8545-451b-a8a0-1929c0afa003)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
