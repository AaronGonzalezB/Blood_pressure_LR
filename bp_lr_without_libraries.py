"""
Linear regression algorithm without ML libraries - Blood pressure by age prediction

- Cost function: measure performance of the LR:

Median Square Error (MSE): Averages all distances
between real points and the LR approximation
(yi - y'i) where yi are the actual points and y'i are the approximation points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Linear model equation
def calcular_modelo(w,b,x):

    return w*x + b

# MSE
def calcular_error(y,y_):

    N = y.shape[0]
    error = np.sum((y-y_)**2)/N
    return error

# Gradient descent
def gradiente_descendente(w_, b_, alpha, x, y):

    N = x.shape[0] 

    # Gradients
    dw = -(2/N)*np.sum(x*(y-(w_*x+b_)))
    db = -(2/N)*np.sum(y-(w_*x+b_))

    # Load new weights
    w = w_ - alpha*dw
    b = b_ - alpha*db

    return w, b

# Data analysis
datos = pd.read_csv('dataset.csv', sep=",", skiprows=32, usecols=[2,3])
print(datos)

datos.plot.scatter(x='Age', y='Systolic blood pressure')
plt.xlabel('Edad (años)')
plt.ylabel('Presión sistólica (mm de Mercurio)')
plt.show()

x = datos['Age'].values
y = datos['Systolic blood pressure'].values

# Learning w and b coefficients with gradient descent

w = np.random.randn(1)[0]
b = np.random.randn(1)[0]

alpha = 0.0004  # learning rate
nits = 40000    # epochs

# Train
error = np.zeros((nits,1))
for i in range(nits):

    [w, b] = gradiente_descendente(w,b,alpha,x,y)

    # Prediction values
    y_ = calcular_modelo(w,b,x)

    # Error values
    error[i] = calcular_error(y,y_)

    if (i+1)%1000 == 0:
        print("Epoch {}".format(i+1))
        print("    w: {:.1f}".format(w), " b: {:.1f}".format(b))
        print("    error: {}".format(error[i]))
        print("=======================================") 

# Graph results - Error aproximation and model
plt.subplot(1,2,1)
plt.plot(range(nits),error)
plt.xlabel('epoch')
plt.ylabel('MSE')

y_regr = calcular_modelo(w,b,x)
plt.subplot(1,2,2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Prediction
edad = 90
presion = calcular_modelo(w,b,edad)
print("A los {}".format(edad), " anios se tendra una presion sanguinea de {:.1f}".format(presion))
