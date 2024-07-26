import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Datos históricos de consumo de energía en Venezuela
datos = np.array([
    [2011, 91.185],
    [2012, 93.878],
    [2013, 95.128],
    [2014, 56.786],
    [2015, 86.153],
    [2016, 82.283],
    [2017, 70.493],
    [2018, 67.749],
    [2019, 57.844],
    [2020, 54.362],
    [2021, 56.656],
    [2022, 56.768]
], dtype=float)

# Separar los datos en años y valores de consumo
años = datos[:, 0].reshape(-1, 1)
consumo = datos[:, 1]

# Normalizar los datos de los años
escalador = StandardScaler()
años_normalizados = escalador.fit_transform(años)

# Construcción de la red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1)
])

# Compilación del modelo con el optimizador y la función de pérdida
modelo.compile(loss='mean_squared_error', optimizer='adam')

# Entrenamiento del modelo con los datos normalizados
modelo.fit(años_normalizados, consumo, epochs=2000)

# Función para predecir el consumo de energía en un año determinado
def predecir_consumo(año):
    año_pre = escalador.transform([[año]])
    return modelo.predict(año_pre)[0][0]

# Función para predecir y escalar años futuros
def predecir_y_escalar(años, escalador, modelo):
    A_norm = escalador.transform(años.reshape(-1, 1))
    return modelo.predict(A_norm)

# Predicción para los años futuros (2015-2027)
futuro = np.arange(2015, 2028)
predicciones_futuras = predecir_y_escalar(futuro, escalador, modelo)

# Graficar los datos históricos y las predicciones futuras
plt.plot(años, consumo, 'bo-', label='Datos')
plt.plot(futuro, predicciones_futuras, 'ro-', label='Predicción')
plt.xlabel('Año')
plt.ylabel('Consumo de Energía (GW)')
plt.title('Predicción del Consumo de Energía en Venezuela')
plt.legend()
plt.show()
