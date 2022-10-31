# Estimacion de la serie temporal de turistas en Bolzano (Italia) en base a modelos de Redes Neuronales.
# Basado en el trabajo que realice con un modelo SARIMA en el estudio "A dynamic panel data study of the German
# demand for tourism in South Tyrol", Tourism and Hospitality Research, 9(4), 305-313, escrito con Brida, J. G. (2009) 

# Importacion de los programas usuales para gráficas y cálculos
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

# Programas para hacer estimaciones de redes neuronales
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Cargar el set de datos. En este caso una serie mensual de presencias en Bolzano. Solo la columna 1 con los datos
dataframe = read_csv('tirol.csv', usecols=[1])
dataframe.plot(figsize=(10,5),title="Evolucion")

# Convierte valores a float
dataset = dataframe.values
dataset = dataset.astype('float32') 

# Se usa sigmoid y tanh que son sensibles a la magnitud, por lo cual los valores necesitan estar normalizados
# Esto normaliza el set de datos
scaler = MinMaxScaler(feature_range=(0, 1)) #Reescala toda al rango de 0 a 1
dataset = scaler.fit_transform(dataset)

# No se puede usar una manera aleatorioa de dividir el set de datos en entrenamiento (train) y test (test) porque la secuencia de 
# eventos es importante en series de tiempos. Tomemos 70% de los valores para entrenamiento y el resto para testear el set
# Se divide el set en Entrenamiento y test
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# No se puede ajustar el modelo como se hace normalmente con el procesamiento de imagenes donde hay X y Y. Se necesita transformar 
# los datos en algo que parezca valores de X y Y. De esta manera se pueden entrenar en secuencia en lugar de datos puntuales.
# Se convierten en n numeros de columnas para X donde se alimenta la secuencia de numeros entonces la columna final como Y donde
# se le prove el siguiente numero en la secuencia como salida (output). Convirtamos en un array de valores en matriz de datos

# seq_size es el numero de pasos temporales previos a usar como variables insumos para predecir el siguiente periodo.
# Se crea in set de dats donde X es el numero de PIB en los tiempos (t, t-1, t-2,...) y Y es el numero de PIN en el momento 
# (t+1)

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size):  #corrección quitando el -1
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)

seq_size = 12  # Numero de pasos temporales para mirar atrás 

trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

num_features = 1 # Ejemplo univariado

##################################################################
# MODELO  LSTM Simple

#  Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#print('Single LSTM with hidden Dense...')
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(None, seq_size)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
 
model.summary()
print('Train...')

###################################################################

# Entrenamiento del Modelo
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=400)

# Hacer predicciones para validar 
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict.shape
# (483, 1)

# Invertir predicciones a los valores preescalados
# Esto es para comparar con los valores originales de entrada
# Como se uso minmaxscaler ahora se puede usar scaler.inverse_transform para invertir la transformación
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calcular la raíz de los errores cuadráticos medios (msr)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Train Score: 155891.87 RMSE
# Test Score: 215023.74 RMSE

####### cambiar los datos de entrenamiento predichos para graficar ######

# se necesita cambiar (shift) las predicciones de forma que se alineen en el eje de las x con la serie original
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2):len(dataset), :] = testPredict

fig, axe = plt.subplots(figsize=(16, 8))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
#########################################################################

## PREDICCION HACIA ADELANTE ############################################

#predicción
prediction = [] # lista vacía para acumular las predicciones
current_batch = test[-seq_size:] # Últimos datos observados de la serie para comenzar la prediccion hacia adelante
current_batch = current_batch.reshape(1, num_features, seq_size) # Reshape

## Mese a predecir 
future = 24 # Meses a predecir
for i in range(len(test),len(test) + future):
    current_pred = model.predict(current_batch)[0]
    prediction.append(current_pred)
    current_batch = np.append(current_batch[:,:,1:],current_pred)
    current_batch=np.reshape(current_batch, (1, 1, current_batch.shape[0]))

rescaled_prediction = scaler.inverse_transform(prediction)

### DEVOLVER LA LISTA DE PREDICCIONES

list(rescaled_prediction.flatten())

########################################################################################################################