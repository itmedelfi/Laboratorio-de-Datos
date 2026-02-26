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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#%% Carga de datos


df_letras = pd.read_csv('TP02-EnglishTypeAlphabet.csv')

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



#%% Ejercicio 2: Clasificación binaria

# Dataframe con solo el subconjunto de imágenes correspondientes a las letras O y L
df_L_O = df_letras[(df_letras["label"] == 11) | (df_letras["label"] == 14)] 

# Separamos en las imágenes correspondientes a la letra L
df_L = df_letras[(df_letras["label"] == 11)]

# Separamos en las imágenes correspondientes a la letra O
df_O = df_letras[(df_letras["label"] == 14)]

# Calculamos la media de cada columna
media_L = df_L.iloc[:,1:].mean()
media_O = df_O.iloc[:,1:].mean()


# Calculamos la diferencia entre la media de las imágenes correspondientes a las letras O y L
diferencia = media_L - media_O
diferencia_img = diferencia.values.reshape(28,28)
interseccion_img = np.minimum(media_L.values, media_O.values).reshape(28,28)

# Graficamos la diferencia usando un mapa de color.
im = plt.imshow(diferencia_img, cmap='RdBu', vmin=-abs(diferencia).max(), vmax=abs(diferencia).max())
plt.title("Intensidad Media por Píxel(L-O)", fontsize=14)
plt.colorbar(im)
plt.axis('off')
plt.show()

# Graficamos la interseccion
plt.figure()
plt.imshow(interseccion_img, cmap='magma_r')
plt.title("Intersección de trazos", fontsize=14)
plt.colorbar(label='Nivel de Intensidad')
plt.axis('off')
plt.show()


# Separamos los datos y las etiquetas
X1 = df_L_O.iloc[:, 1:]
y1 = df_L_O["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42)

top12 = diferencia.abs().sort_values(ascending=False).head(12)

def ajuste (indices, k):
    X_tr = X_train.iloc[:,indices]
    X_ts = X_test.iloc[:,indices]
    
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_tr, y_train)
    
    y_prd_ts = modelo.predict(X_ts)
    
    acc = accuracy_score(y_test, y_prd_ts)
    
    return (f"Atributos usados: {indices}, Accuracy: {acc}")
    

# Seleccionamos los siguientes 3 atributos con mayor diferencia, con un k = 3
print(ajuste([245,273,453], 3))

print(ajuste([425,358,330], 3))

print(ajuste([245,273,453], 10))
#%%

#%% --- representacion grafica
# img = np.array(df_letras.sample()).reshape((28,28))
# plt.imshow(img, cmap='gray')
# plt.show()

#%% --- funciones propias


#%% --- código fuera de las funciones