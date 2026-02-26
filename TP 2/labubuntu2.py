""" 
Laboratorio de Datos - Verano 2026

Integrantes:
- Lanabere, Delfina Daniela (LU: 246/24)
- Muhafra, Micaela Abril (LU: 1327/24)
- Gomez Arreaza, Catherine De Jesus (LU: 980/24)

Datos relevantes:
"""

#%% Importación de librerias

import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#%% Carga de datos

carpeta = r'C:\Users\Delfina\Desktop\EXACTAS\LABO_DATOS\tp2'
df_letras = pd.read_csv(carpeta + '\TP02-EnglishTypeAlphabet.csv')

#%% Visualización de letra por índice

def visualizar_letra(indice):
    # se extrae de la fila la columna label para luego usar
    # .iloc[indice, 1:] para saltar la etiqueta que esta en la columna 0
    pixeles = df_letras.iloc[indice, 1:].values
    
    # luego se redimensiona a 28x28
    matriz = pixeles.reshape(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(matriz, cmap='gray')
    plt.axis('off')
    
    plt.show()

visualizar_letra(0)

#%% Ejercicio 1: Análisis exploratorio

# Cantidad de datos y atributos
print(f"Dimensiones del dataset: {df_letras.shape}")
# 26416 datos con 785 atributos c/u

# Cantidad de clases (letras)
clases = df_letras['label'].unique()
print(f"Cantidad de clases: {len(clases)}")
# 26 clases, una por cada letra del abecedario (menos la ñ)

# Distribución de las clases
distribucion = df_letras['label'].value_counts().sort_index()
distribucion.plot(kind='bar', figsize=(10, 4))
plt.title("Cantidad de muestras por letra")
plt.show()
print(f"Cant. de muestras por cada clase: {26416/26}")
# en el dataset se muestra una misma cantidad de distintas clases
# por ende podemos intuir que, al estar perfectamente balanceado
# se cuenta con 1016 observaciones por cada una de las 26 clases
# de la variable de interés



media_pixeles = df_letras.iloc[:, 1:].mean().values.reshape(28, 28)
plt.figure(figsize=(7, 6))
# calculamos la media de cada píxel en todo el dataset
# para ver qué atributos importan más

# Visualizamos la media usando un mapa de color
im = plt.imshow(media_pixeles, cmap='Blues')
plt.title("Intensidad Media por Píxel", fontsize=14)
plt.colorbar(im, label='Nivel de intensidad promedio')
plt.axis('off')

plt.show()



# oObtenemos las letras únicas y las ordenamos
letras = sorted(df_letras['label'].unique())

plt.figure(figsize=(22, 7))

# Usamos un for para recorrer cada letra
for i, letra in enumerate(letras):
    plt.subplot(3, 13, i+1) # Grilla para las 26 letras
    
    # Filtramos las filas de esa letra y calculamos la media de cada columna
    img_promedio = df_letras[df_letras['label'] == letra].iloc[:, 1:].mean().values.reshape(28, 28)
    
    plt.imshow(img_promedio, cmap='gray')
    plt.title(letra)
    plt.axis('off')

plt.tight_layout()
plt.suptitle('Imágenes de la media de cada clase', fontsize=25, y=1.02)
plt.show()



# Seleccionamos 10 muestras aleatorias de la letra J
muestras_j = df_letras[df_letras['label'] == 9].sample(10, random_state=42)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = muestras_j.iloc[i, 1:].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()


#%% --------------------------------------------------------------------------------------
# Clasificación multiclase (EJ: 3)
# ----------------------------------------------------------------------------------------
# Dada una imagen se desea responder la siguiente pregunta: ¿A cuál de las clases corresponde la imagen?  
"""
c. Realizar un experimento para comparar y seleccionar distintos árboles de decisión, con distintos hiperparámetos. 
    Limitarse a usar profundidades entre 1 y 10. 
    Para esto, utilizar validación cruzada con k-folding. 
    ¿Cuál fue el mejor modelo? Documentar cuál configuración de hiperparámetros es la mejor, y qué performance tiene. 
d. Entrenar el modelo elegido a partir del inciso previo, ahora en todo el conjunto de desarrollo. 
    Utilizarlo para predecir las letras del conjunto held-out y reportar la performance.  """
#%% 3.a Separar el conjunto de datos

# Armamos la lista "clases" la posicion de cada letra corresponde al label del dataframe
clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#dividimos el dataframe en desarrollo dev (85%) y validación heldout (15%)
X_dev, X_eval, y_dev, y_eval = train_test_split(
        X,y, test_size = 0.15, stratify = y, random_state = 20)

#%% 3.b Ajustar un modelo de árbol de decisión (Profundidad 1 a 20)

profundidades_3b = [1,3,5,8,11,14,17,20]
#armamos las variables donde vamos a guardar como le fue a cada arbol
scores_train = [] #evaluacion sobre letras ya vistas
scores_test = []  #evaluacion sobre letras no vistas

for d in profundidades_3b: #d es la cantidad de nodos
    arbol_b = tree.DecisionTreeClassifier(max_depth=d, random_state=42)
    # Separamos en train (80%) y test (20%)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_dev, y_dev, test_size=0.2)
    # Estudiamos las imágenes
    arbol_b.fit(X_train_b, y_train_b)
    
    #Guardamos el puntaje del arbol
    scores_train.append(arbol_b.score(X_train_b, y_train_b))
    scores_test.append(arbol_b.score(X_test_b, y_test_b))
#%%
# Graficamos el resultado de 3.b para poder visualizar como cambia
plt.figure(figsize=(8, 5))
plt.plot(profundidades_3b, np.array(scores_train)*100, label='Puntaje sobre Train', marker='o')
plt.plot(profundidades_3b, np.array(scores_test)*100, label='Puntaje sobre Test', marker='o')
plt.title("3.b: Evolución de la Exactitud según Profundidad del Árbol(1 a 20)")
plt.xlabel("Profundidad")
plt.xticks(profundidades_3b)
plt.ylabel("Exactitud (%)")
plt.grid(True, linestyle='--', alpha=0.6) # Agregamos una grilla para leerlo mejor
plt.legend()
plt.show()

# %% 3.c Experimento y comparación de árboles

profundidades = [1, 3, 5, 8, 10]
nsplits = 5 # Dividimos los datos en 5
kf = KFold(n_splits=nsplits)

# Creamos una matriz para guardar la exactitud de cada arbol. Tiene 5 filas (una por cada fold) y 10 columnas (una por cada profundidad)
resultados = np.zeros((nsplits, len(profundidades)))


for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    # Separamos los datos para train y test
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, hmax in enumerate(profundidades):
        # Para cada rpofundidad entrenamos el modelo con los datos "train" y lo probamos con los de "test"
        arbol = tree.DecisionTreeClassifier(max_depth=hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        score = accuracy_score(kf_y_test, pred)

        #guardamos el resultado en la tabla
        resultados[i, j] = score
        
#%% 
# Promediamos los scores sobre para cada altura para quedarnos con el modelo que mejor rindió en todos los tests
scores_promedio = resultados.mean(axis = 0)

# Buscamos y guradamos el promedio más alto
mejor_profundidad = profundidades[np.argmax(scores_promedio)]

#%% 
for i,e in enumerate(profundidades):
    print(f'Score promedio del modelo con hmax = {e}: {scores_promedio[i]:.4f}')

# %% entreno el modelo elegido en el conjunto dev entero
arbol_elegido = tree.DecisionTreeClassifier(max_depth=10)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_dev)

score_arbol_elegido_dev = accuracy_score(y_dev, y_pred)
print(score_arbol_elegido_dev)

# %% pruebo el modelo y se entrana con en el conjunto eval
y_pred_eval = arbol_elegido.predict(X_eval)
score_arbol_elegido_eval = accuracy_score(y_eval, y_pred_eval)
print(score_arbol_elegido_eval)
# %%
# Calculo mis predicciones
y_pred = arbol_elegido.predict(X_eval)

# Computo la matriz de confusión comparando y con y_pred
matriz = confusion_matrix(y_eval, y_pred)
print("Matriz de confusión:")
print(matriz)

# Computo la exactitud comparando y con y_pred
accuracy = accuracy_score(y_eval, y_pred)
print("Exactitud:", accuracy)

#Grafico la matriz de confusion
plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot(cmap='Greens',colorbar=False,ax=plt.gca())
plt.title("Matriz de confusión (hmax = 10)")
plt.show()


