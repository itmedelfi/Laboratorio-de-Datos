""" 
Laboratorio de Datos - Verano 2026

Integrantes:
- Lanabere, Delfina Daniela (LU: 246/24)
- Muhafra, Micaela Abril (LU: 1327/24)
- Gomez Arreaza, Catherine De Jesus (LU: 980/24)
"""

#%% Importación de librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

#%% Carga de datos

# directorio donde se encuentra el csv del alfabeto
carpeta = r'C:\Users\~'
df_letras = pd.read_csv(carpeta + '\TP02-EnglishTypeAlphabet.csv')

#%% Separación de Variables

# X variable explicativa: para las imagenes son los valores de los pixeles
# Y variable a explicar: para las clases (la letra)
X = df_letras.drop("label", axis=1)
y = df_letras["label"]


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



# Obtenemos las letras únicas y las ordenamos
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

# Separamos los datos y las etiquetas
X1 = df_L_O.iloc[:, 1:]
y1 = df_L_O["label"]

# Generamos nuestros conjuntos de training y testing.
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42)

# Reconstruimos el dataframe con los labels para poder separar por letra.
df_train = X_train.copy()
df_train["label"] = y_train

# Separamos en las imágenes correspondientes a la letra L
df_L = df_train[df_train["label"] == 11]

# Separamos en las imágenes correspondientes a la letra O
df_O = df_train[df_train["label"] == 14]

# Calculamos la media de cada columna(píxel)
media_L = df_L.iloc[:,:-1].mean()
media_O = df_O.iloc[:,:-1].mean()

# Calculamos la varianza de cada columna(píxel)
var_L = df_L.iloc[:,:-1].var()
var_O = df_O.iloc[:,:-1].var()

# Calculamos la diferencia entre la media de las imágenes correspondientes a las letras O y L
diferencia = media_L - media_O

# Buscamos píxeles que sean consistentes en dentro de los dataframes, pero que varien entre ellos.
S = (media_L - media_O)**2 / (var_L + var_O)

# Creamos distintas listas de indices para utilizar en el training del modelo y comparar sus resultados entre sí.
top2_dif = diferencia.abs().sort_values(ascending=False).index[:2].tolist()
top6_dif = diferencia.abs().sort_values(ascending=False).index[:6].tolist()
top12_dif = diferencia.abs().sort_values(ascending=False).index[:12].tolist()
top25_dif = diferencia.abs().sort_values(ascending=False).index[:25].tolist()
top65_dif = diferencia.abs().sort_values(ascending=False).index[:65].tolist()
top2_fs = S.sort_values(ascending=False).index[:2].tolist()
top6_fs = S.sort_values(ascending=False).index[:6].tolist()
top12_fs = S.sort_values(ascending=False).index[:12].tolist()
top25_fs = S.sort_values(ascending=False).index[:25].tolist()
top65_fs = S.sort_values(ascending=False).index[:65].tolist()
numeros_azar = np.random.choice(range(784), size=6, replace=False)
columnas_azar2 = X_train.columns[numeros_azar[:2]].tolist()
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
    ("diferencia_media", 2, top2_dif),
    ("azar", 2, columnas_azar2),
    ("fisher_score", 2, top2_fs),
    ("borde", 2, ['pixel 0', 'pixel 1']),
    
    ("diferencia_media", 6, top6_dif),
    ("azar", 6, columnas_azar6),
    ("fisher_score", 6, top6_fs),
    ("borde", 6, ['pixel 0','pixel 1','pixel 2','pixel 3','pixel 4','pixel 5']),
    
    ("diferencia_media", 12, top12_dif),
    ("fisher_score", 12, top12_fs),
        
    ("diferencia_media", 25, top25_dif),
    ("fisher_score", 25, top25_fs),
    
    ("diferencia_media", 65, top65_dif),
    ("fisher_score", 65, top65_fs),

]

valores_k = [2, 5, 35]

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

#%% Ejercicio 3: Clasificación multiclase
# Dada una imagen se desea responder la siguiente pregunta: ¿A cuál de las clases corresponde la imagen?   

#%% Funciones auxiliares del punto 3
def experimentar_profundidad(X_datos, y_datos, profundidades):
    """
    Entrena árboles con distintas profundidades y devuelve los scores.
    """
    train_scores = []
    test_scores = []
    
    for p in profundidades:
        # Definimos el modelo
        arbol = tree.DecisionTreeClassifier(max_depth=p, random_state=42)
        
        # Separamos datos (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X_datos, y_datos, test_size=0.2)
        
        # Entrenamos y evaluamos
        arbol.fit(X_train, y_train)
        train_scores.append(arbol.score(X_train, y_train))
        test_scores.append(arbol.score(X_test, y_test))
        
    return np.array(train_scores), np.array(test_scores)

def graficar_resultados(profundidades, scores_train, scores_test, titulo):
    """
    Genera el gráfico de evolución de exactitud.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(profundidades, scores_train * 100, label='Puntaje sobre Train', marker='o')
    plt.plot(profundidades, scores_test * 100, label='Puntaje sobre Test', marker='o')
    
    plt.title(titulo)
    plt.xlabel("Profundidad")
    plt.ylabel("Exactitud (%)")
    plt.xticks(profundidades)
    plt.ylim(0, 105) # Un poquito más de 100 para que no se corte el punto
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
#%% 3.a Separar el conjunto de datos

# Armamos la lista "clases" la posicion de cada letra corresponde al label del dataframe
letras = [chr(i) for i in range(65, 91)]

#dividimos el dataframe en desarrollo dev (85%) y validación heldout (15%)
X_dev, X_eval, y_dev, y_eval = train_test_split(
        X,y, test_size = 0.15, stratify = y, random_state = 42)

#%% 3.b1 Ajustar un modelo de árbol de decisión (Profundidad 1 a 20 salteado)
profs_1 = [1, 4, 9, 13, 17, 20]
s_train_1, s_test_1 = experimentar_profundidad(X_dev, y_dev, profs_1)
graficar_resultados(profs_1, s_train_1, s_test_1, "3.b: Precisión del arbol de desiciones segun nodos")

#%% 3.b2 Detalle del modelo entre nodos 9 y 17

profs_2 = np.arange(9, 18)
s_train_2, s_test_2 = experimentar_profundidad(X_dev, y_dev, profs_2)
graficar_resultados(profs_2, s_train_2, s_test_2, "3.b: Detalle en alturas medias (9 a 17)")

#Guardamos la mejor altura (leer el informe para más información)
mejor_altura = 11

# %% 3.c Experimento y comparación de árboles
#Los hiperparámetros a evaluar son el los criterios de las medidas de impureza y las profundidades maximas

# Definimos los rangos de hiperparámetros
criterios = ['gini', 'entropy']
profundidades_3c = [1,3,5,8,10]
nsplits = 5 # Dividimos los datos en 5
kf = KFold(n_splits=nsplits)

# Diccionario para guardar resultados de cada criterio
resultados = {crit: np.zeros((nsplits, len(profundidades_3c))) for crit in criterios}

# para cada criterio
for crit in criterios:
    print(f"Evaluando criterio: {crit}...")
    # para cada fold
    for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    
        # Separamos los datos para train y test
        kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
        # para cada altura
        for j, d in enumerate(profundidades_3c):
            # Para cada rpofundidad entrenamos el modelo con los datos "train" y lo probamos con los de "test"
            # Creamos el árbol con la combinación actual de hiperparámetros
            arbol = tree.DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=42)
            arbol.fit(kf_X_train, kf_y_train)
            
            # Evaluación
            pred = arbol.predict(kf_X_test)
            resultados[crit][i, j] = accuracy_score(kf_y_test, pred)
         
# Promediamos los scores para cada altura y criterio para quedarnos con el modelo que mejor rindió en todos los tests
promedios_gini = resultados['gini'].mean(axis=0)
promedios_entropy = resultados['entropy'].mean(axis=0)

# Buescamos el ganador absoluto
if np.max(promedios_gini) >= np.max(promedios_entropy):
    mejor_crit = 'gini'
    mejor_h = profundidades_3c[np.argmax(promedios_gini)]
    mejor_score = np.max(promedios_gini)
else:
    mejor_crit = 'entropy'
    mejor_h = profundidades_3c[np.argmax(promedios_entropy)]
    mejor_score = np.max(promedios_entropy)
#%%
# Imprimimos todos los resultados
print(f"Mejor configuración: Criterio = {mejor_crit}, Profundidad = {mejor_h}")
print(f"Performance promedio en CV: {mejor_score:.4f}")

#%% 3.c

# Graficamos la línea de Gini
plt.plot(profundidades_3c, promedios_gini * 100, label='Criterio: Gini', 
         marker='o', linestyle='-', linewidth=2, color='purple')

# Graficamos la línea de Entropía
plt.plot(profundidades_3c, promedios_entropy * 100, label='Criterio: Entropía', 
         marker='s', linestyle='--', linewidth=2, color='darkcyan')

plt.title("3.c: Comparación de Hiperparámetros mediante k-folding", fontsize=14)
plt.xlabel("Profundidad Máxima ($h_{max}$)", fontsize=12)
plt.xticks(profundidades_3c)
plt.ylabel("Exactitud Promedio (%)", fontsize=12)
plt.ylim(0, 105) # Un poquito más de 100 para que no se corte el punto
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.show()

# %% 3.d Entrenamos el modelo elegido en el conjunto dev entero
mejor_criterio = mejor_crit         # La mejor medida de impureza
mejor_profundidad = mejor_altura    # La mejor profundidad

arbol_elegido = tree.DecisionTreeClassifier(
    criterion=mejor_criterio, 
    max_depth=mejor_profundidad, 
    random_state=42
)
# Entrenamos todo el conjunto de desarrollo
arbol_elegido.fit(X_dev, y_dev)

#Predecimos sobre el conjunto Held-Out separado en el punto 3.a
y_pred = arbol_elegido.predict(X_eval)

# Evaluación Final
exactitud_arbol_dev = accuracy_score(y_eval, y_pred)
print("Exactitud: ", exactitud_arbol_dev*100)

# %% 
# Armamos la matriz de confusión (Usando la predicción que ya hicimos arriba)
matriz = confusion_matrix(y_eval, y_pred)

plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=letras)
disp.plot(cmap='PuBu', colorbar=False, ax=plt.gca(), values_format='d')

plt.title(f"Matriz de confusión (hmax = {mejor_altura}, crit = {mejor_crit})", fontsize = 14)
plt.xlabel("Letra Predicha", fontsize=12)
plt.ylabel("Letra Real", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

#%%
#Identificamos los extremos (letras con mejor y peor performance)
aciertos_por_letra = matriz.diagonal()

# Letra con más aciertos
idx_max = np.argmax(aciertos_por_letra)
letra_max = letras[idx_max]
cant_max = aciertos_por_letra[idx_max]

# Letra con menos aciertos
idx_min = np.argmin(aciertos_por_letra)
letra_min = letras[idx_min]
cant_min = aciertos_por_letra[idx_min]

print(f"Letra con MÁS aciertos: {letra_max} ({cant_max})")
print(f"Letra con MENOS aciertos: {letra_min} ({cant_min})")

#%%
# Identificamos los pixeles mas importantes para la toma de decisiones en un vector de 784 elementos
importancias = (arbol_elegido.feature_importances_)*100

#armamos la matriz de 28x28 desde el vector
importancia_imagen = importancias.reshape(28, 28)

#Graficamos el mapa de color
plt.figure(figsize=(8, 6))
plt.imshow(importancia_imagen, cmap='YlOrRd', interpolation='nearest')

# Detalles del gráfico
plt.colorbar(label='Grado de Importancia de los píxeles')
plt.title("Píxeles Críticos para la Decisión", fontsize=14)
plt.axis('off') # Quitamos los ejes porque representan la forma de la letra
plt.show()

