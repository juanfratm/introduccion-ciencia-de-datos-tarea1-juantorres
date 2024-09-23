# id:17--34-17

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



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
#print('Train Set: ', x_train.shape, y_train.shape)
#print('Test Set: ', x_test.shape, y_test.shape)

# Entrenar el modelo de regresión logística
LR = LogisticRegression()
LR.fit(x_train, y_train)

# Imprimir los coeficientes y el intercepto
#print('The slopes are: ', LR.coef_[0])
#print('The intercept is: ', LR.intercept_)

##predicciones
predictions = LR.predict(x_train)
score = LR.score(x_train, y_train)
#print('The score is: ', score)

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



# Valores para el parámetro C
valores_C = [0.001, 1, 100]

# Definir un nuevo nombre para los valores x que se usarán para trazar la frontera de decisión
valores_x_svm = np.linspace(X1.min(), X1.max(), 100)

# ENTRENAMIENTO Y EVALUACIÓN DEL MODELO SVM PARA DIFERENTES VALORES DE C
for valor_C in valores_C:
    # Usar dual=False cuando el número de muestras es mayor que el número de características
    modelo_svm = LinearSVC(max_iter=10000, C=valor_C, verbose=1, dual=False)
    modelo_svm.fit(x_train, y_train)

    # Predicciones en los datos de entrenamiento
    predicciones_svm = modelo_svm.predict(x_train)

    # Puntaje del modelo
    puntaje_svm = modelo_svm.score(x_train, y_train)
    print(f"Puntaje del SVM con C={valor_C}: ", puntaje_svm)

    # Obtener coeficientes e intercepto para trazar la frontera de decisión
    print(f"Coeficientes del SVM con C={valor_C}: {modelo_svm.coef_}")
    print(f"Intercepto del SVM con C={valor_C}: {modelo_svm.intercept_}")

    coeficiente_svm = modelo_svm.coef_[0]
    intercepto_svm = modelo_svm.intercept_

    # Calcular la frontera de decisión para el SVM
    frontera_decision_svm = -(coeficiente_svm[0] / coeficiente_svm[1]) * valores_x_svm - intercepto_svm / coeficiente_svm[1]

    # Graficar los datos originales, las predicciones y la frontera de decisión
    plt.scatter(X1[Y == -1], X2[Y == -1], c='black', marker='o', label='-1 (Original)', s=40)
    plt.scatter(X1[Y == 1], X2[Y == 1], c='red', marker='+', label='+1 (Original)', s=70)
    plt.scatter(x_train[predicciones_svm == -1, 0], x_train[predicciones_svm == -1, 1], c='blue', marker='x', label='-1 (Predicción)', s=50)
    plt.scatter(x_train[predicciones_svm == 1, 0], x_train[predicciones_svm == 1, 1], c='orange', marker='s', label='+1 (Predicción)', s=50)

    # Graficar la frontera de decisión
    plt.plot(valores_x_svm, frontera_decision_svm, label=f'Frontera de decisión (SVM, C={valor_C})', color='green', linestyle='--', linewidth=5)
    plt.xlabel('x_1', fontsize=14)
    plt.ylabel('x_2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.15, 1.15), loc='upper right', fancybox=True, framealpha=1, fontsize=12)
    plt.title(f'Predicciones y Frontera de Decisión (SVM, C={valor_C})')
    plt.show()

    # Guardar la imagen con el valor de C
    plt.savefig(f'predicciones_frontera_decision_C_{valor_C}.png')

    # Matriz de confusión y reporte de clasificación
    matriz_confusion_svm = confusion_matrix(y_train, predicciones_svm)
    print("Matriz de Confusión para SVM:")
    print(matriz_confusion_svm)
    print("Reporte de Clasificación para SVM:")
    print(classification_report(y_train, predicciones_svm))

    # Graficar la matriz de confusión para SVM
    plt.figure(figsize=(6, 6))
    sns.heatmap(matriz_confusion_svm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.title(f'Matriz de Confusión para SVM (C={valor_C})', size=15)
    plt.savefig(f'Figura_SVM_C_{valor_C}.png')
    plt.show()

