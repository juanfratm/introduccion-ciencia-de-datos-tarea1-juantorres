# id:17--34-17

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


df=pd.read_csv('C:/Users/HP/Documents/TAREA INTRODUCCION A CIENCIA DE DATOS/TAREA/data/dataset1.csv')

###print(df.isnull().sum()) ###Verificamos Valores Nulos

df.columns = ['X1', 'X2', 'Y']

#print(df.head()) ###Exploramos la data

# Definir X1 y X2
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]

# Combinar X1 y X2
X = np.column_stack((X1, X2))

# Definir la variable de respuesta
Y = df.iloc[:, 2]

# Grafica de los datos del dataset
plt.scatter(X1[Y == -1], X2[Y == -1], c='k', marker='o', label='-1', s=40)
plt.scatter(X1[Y == 1], X2[Y == 1], c='r', marker='+', label='+1', s=70)

plt.xlabel('x_1', fontsize=14)
plt.ylabel('x_2', fontsize=14)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#Leyenda de la Gráfica
plt.legend(bbox_to_anchor=(1.15, 1.15), loc='upper right', fancybox=True, framealpha=1, fontsize=12)

plt.savefig('Figure_1.png')

plt.show()

###ENTRENAMIENTO DEL MODELO

# Separar en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=23)
print('Train Set: ', x_train.shape, y_train.shape)
print('Test Set: ', x_test.shape, y_test.shape)

# Entrenar el modelo de regresión logística
LR = LogisticRegression()
LR.fit(x_train, y_train)

# Imprimir los coeficientes y el intercepto
print('The slopes are: ', LR.coef_[0])
print('The intercept is: ', LR.intercept_)

##predicciones
predictions = LR.predict(x_train)
score = LR.score(x_train, y_train)
print('The score is: ', score)

# Coeficientes e intercepto del modelo de regresión logística
coeficiente = LR.coef_[0]
intercepto = LR.intercept_

# Valores para x1 (eje x) para la frontera de decisión
x_values = np.linspace(X1.min(), X1.max(), 100)

# Frontera de decisión
frontera_decision = -(coeficiente[0] / coeficiente[1]) * x_values - intercepto / coeficiente[1]

# Gráfica con los datos del dataset
plt.scatter(X1[Y == -1], X2[Y == -1], c='black', marker='o', label='-1 (Original)', s=40)
plt.scatter(X1[Y == 1], X2[Y == 1], c='red', marker='+', label='+1 (Original)', s=70)

# Gráfica con las predicciones
plt.scatter(x_train[predictions == -1, 0], x_train[predictions == -1, 1], c='blue', marker='x', label='-1 (Predicción)', s=50)
plt.scatter(x_train[predictions == 1, 0], x_train[predictions == 1, 1], c='orange', marker='s', label='+1 (Predicción)', s=50)

# Grafica de la frontera de decisión
plt.plot(x_values, frontera_decision, label='Frontera de decisión', color='purple', linestyle='--', linewidth=5)

# Etiquetas y detalles del gráfico
plt.xlabel('x_1', fontsize=14)
plt.ylabel('x_2', fontsize=14)
plt.legend(bbox_to_anchor=(1.15, 1.15), loc='upper right', fancybox=True, framealpha=1, fontsize=12)

# Guardar y mostrar el gráfico
plt.savefig('Figure-2.png')
plt.show()


