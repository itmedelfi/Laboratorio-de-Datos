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

# Fijamos una semilla, para que los números generados al azar sean siempre los mismos, y tener consistencia en el 
# análisis.
np.random.seed(42)


# Dataframe con solo el subconjunto de imágenes correspondientes a las letras O y L
df_L_O = df_letras[(df_letras["label"] == 11) | (df_letras["label"] == 14)] 

# Separamos en las imágenes correspondientes a la letra L
df_L = df_L_O[df_L_O["label"] == 11]

# Separamos en las imágenes correspondientes a la letra O
df_O = df_L_O[df_L_O["label"] == 14]

# Calculamos la media de cada columna(píxel)
media_L = df_L.iloc[:,1:].mean()
media_O = df_O.iloc[:,1:].mean()

# Calculamos la varianza de cada columna(píxel)
var_L = df_L.iloc[:,1:].var()
var_O = df_O.iloc[:,1:].var()

# Calculamos la diferencia entre la media de las imágenes correspondientes a las letras O y L
diferencia = media_L - media_O

# Buscamos píxeles que sean consistentes en dentro de los dataframes, pero que varien entre ellos.
S = (media_L - media_O)**2 / (var_L + var_O)


# Separamos los datos y las etiquetas
X1 = df_L_O.iloc[:, 1:]
y1 = df_L_O["label"]

# Generamos nuestros conjuntos de training y testing.
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42)

# Creamos distintas listas de indices para utilizar en el training del modelo y comparar sus resultados entre sí.
top3_dif = diferencia.abs().sort_values(ascending=False).index[:3].tolist()
top6_dif = diferencia.abs().sort_values(ascending=False).index[:6].tolist()
top12_dif = diferencia.abs().sort_values(ascending=False).index[:12].tolist()
top25_dif = diferencia.abs().sort_values(ascending=False).index[:25].tolist()
top3_fs = S.sort_values(ascending = False).index[:3].tolist()
top6_fs = S.sort_values(ascending = False).index[:6].tolist()
top12_fs = S.sort_values(ascending = False).index[:12].tolist()
top25_fs = S.sort_values(ascending = False).index[:25].tolist()
numeros_azar = np.random.choice(range(784), size=6, replace=False)
columnas_azar3 = X_train.columns[numeros_azar[:3]].tolist()
columnas_azar6 = X_train.columns[numeros_azar[:6]].tolist()

# Función para entrenar el modelo, que tiene como entrada los atributos y el k a utilizar y de salida, la exactitud.
def ajuste (indices, k):
    X_tr = X_train[indices]
    X_ts = X_test[indices]
    
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_tr, y_train)
    
    y_prd_ts = modelo.predict(X_ts)
    
    acc = accuracy_score(y_test, y_prd_ts)
    
    return acc

  
resultados = []

# Lista con las distintias configuraciones que vamos a usar para entrenar el modelo.
configuraciones = [
    ("diferencia_media", 3, top3_dif),
    ("azar", 3, columnas_azar3),
    ("fisher_score", 3, top3_fs),
    ("borde", 3, ['pixel 0', 'pixel 1', 'pixel 2']),
    
    ("diferencia_media", 6, top6_dif),
    ("azar", 6, columnas_azar6),
    ("fisher_score", 6, top6_fs),
    ("borde", 6, ['pixel 0','pixel 1','pixel 2','pixel 3','pixel 4','pixel 5']),
    
    ("diferencia_media", 12, top12_dif),
    ("fisher_score", 12, top12_fs),
    
    ("diferencia_media", 25, top25_dif),
    ("fisher_score", 25, top25_fs),
]

valores_k = [3, 5, 25]

# Entrenamos el modelo con las configuraciones y lo agregamos a una lista 'resultados'.
for criterio, cantidad, columnas in configuraciones:
    for k in valores_k:
        acc = ajuste(columnas, k)
        resultados.append({
            "criterio": criterio,
            "n_atributos": cantidad,
            "k": k,
            "accuracy": acc
        })

# Pasamos la lista 'resultados' a un dataframe para facilitar su lectura y análisis.
df_resultados = pd.DataFrame(resultados)

# Vamos a pasar nuestros resultados a unos gráficos.
orden_atributos = sorted(df_resultados["n_atributos"].unique())
valores_k = sorted(df_resultados["k"].unique())

for k_val in valores_k:
    
    # Filtramos por k
    df_k = df_resultados[df_resultados["k"] == k_val]
    
    plt.figure()
    
    # Graficamos una línea por criterio
    for criterio in df_k["criterio"].unique():
        
        df_crit = df_k[df_k["criterio"] == criterio]
        df_crit = df_crit.sort_values("n_atributos")
        
        plt.plot(
            df_crit["n_atributos"],
            df_crit["accuracy"],
            marker='o',
            label=criterio
        )
    
    plt.xlabel("Cantidad de atributos")
    plt.ylabel("Accuracy")
    plt.title(f"Desempeño del modelo KNN (k = {k_val})")
    plt.xticks(orden_atributos)
    plt.ylim(0.5, 1)
    plt.legend()
    plt.grid(True)
    plt.show()
#%% --- representacion grafica
# img = np.array(df_letras.sample()).reshape((28,28))
# plt.imshow(img, cmap='gray')
# plt.show()

#%% --- funciones propias


#%% --- código fuera de las funciones
