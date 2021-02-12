# LR with Keras 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# Import data - only two features: systolic blood pressure and age
datos = pd.read_csv('dataset.csv', sep = ",", skiprows = 32, usecols = [2,3])
print(datos)

datos.plot.scatter(x = 'Age', y = 'Systolic blood pressure')
plt.xlabel('Edad (años)')
plt.ylabel('Presión sistólica (mm Hg)')
plt.show()

x = datos['Age'].values
y = datos['Systolic blood pressure'].values

# Model construction with Keras

# - Input layer: each data per epoch
# - Output layer: each LR output

input_dim = 1
output_dim = 1

# Model container
modelo = Sequential()

# Add LR function
modelo.add(Dense(output_dim, input_dim = input_dim, activation = 'linear'))

# Gradient descent with learning rate parameter
sgd = SGD(lr = 0.0004)

# MSE
modelo.compile(loss = 'mse', optimizer = sgd)
modelo.summary()

# Train model
num_epochs = 40000
batch_size = x.shape[0]     # amount of data per epoch

history = modelo.fit(x,y, epochs = num_epochs, batch_size = batch_size, verbose = 0)

# Model output parameters
capas = modelo.layers[0]            
w, b = capas.get_weights() 

print('Parámetros: w = {:.1f}'.format(w[0][0], b[0]))

# Visualize results

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('MSE vs epochs')

y_regr = modelo.predict(x)
plt.subplot(1,2,2)
plt.scatter(x,y)    
plt.plot(x,y_regr,'r') 
plt.xlabel('x')
plt.ylabel('y')
plt.title('Presión sistólica modelada')
plt.show()

# Prediction
x_pred = np.array([90])
y_pred = modelo.predict(x_pred)
print("La presión sanguinea será de {:.1f} mm Hg".format(y_pred[0][0]), " para una persona de {} años".format(x_pred[0]))