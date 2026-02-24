# --- encabezado
""" 
Nombre Grupo: Labubuntu
Integrantes:
    1
    2
    3
Datos relevantes:
    ...
."""
#%% --- importamos librerias
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#%% --- carga de datos
carpeta = '/home/Estudiante/Descargas/TP02-EnglishTypeAlphabet_compressed/'
df_letras = pd.read_csv(carpeta + 'TP02-EnglishTypeAlphabet.csv')

#%% --- representacion grafica
# img = np.array(df_letras.sample()).reshape((28,28))
# plt.imshow(img, cmap='gray')
# plt.show()

#%% --- funciones propias


#%% --- código fuera de las funciones



