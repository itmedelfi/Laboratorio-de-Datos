# Encabezado
""" 
Nombre Grupo: Labubuntu
Integrantes:
    1
    2
    3
Datos relevantes:
    ...
."""
#%% Importación de librerias
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn import tree

#%% Carga de datos
carpeta = r'C:\Users\Delfina\Desktop\EXACTAS\LABO_DATOS\tp2'
df_letras = pd.read_csv(carpeta + r'\TP02-EnglishTypeAlphabet.csv')

#%% Separación de Variables
# X variable explicativa: para las imagenes son los valores de los pixeles
# Y variable a explicar: para las clases (la letra)
X = df_letras.drop("label", axis=1)
y = df_letras["label"]

#%% Visualización de letra por índice

def visualizar_letra(indice):
    # extraer la fila sacando la columna label
    # usar .iloc[indice, 1:] para saltar la etiqueta que esta en la columna 0
    pixeles = df_letras.iloc[indice, 1:].values
    
    # redimensionar a 28x28
    matriz = pixeles.reshape(28, 28)
    
    # graficar
    plt.figure(figsize=(4, 4))
    plt.imshow(matriz, cmap='gray')
    plt.axis('off') # saca los ejes para que se vea mas como una imagen y no tanto como un grafico
    
    plt.show()

visualizar_letra(26415)

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


# Calculamos la varianza de cada píxel (excluyendo la columna label) [cite: 15]

varianzas = df_letras.iloc[:, 1:].var()

# Lo convertimos a matriz 28x28 para verlo como imagen [cite: 21]
mapa_varianza = varianzas.values.reshape(28, 28)

plt.imshow(mapa_varianza, cmap='hot')
plt.colorbar(label='Varianza')
plt.title("Mapa de calor: Relevancia de píxeles por varianza")
plt.show()
# los píxeles oscuros (bordes) tienen varianza ~0 
# y podrían descartarse[cite: 16].

# Análisis de píxeles constantes (Varianza = 0)
# Los píxeles que nunca cambian no aportan información para clasificar
pixeles_constantes = (varianzas == 0).sum()
total_atributos = len(varianzas)

print(f"Píxeles constantes (varianza 0): {pixeles_constantes} de {total_atributos}")

#%% --- representacion grafica
# img = np.array(df_letras.sample()).reshape((28,28))
# plt.imshow(img, cmap='gray')
# plt.show()

#%% --- funciones propias


#%% --- código fuera de las funciones



