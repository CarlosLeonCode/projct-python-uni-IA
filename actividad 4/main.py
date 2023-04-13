import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('dataset.csv')

# Seleccionar las variables dependientes (cantidad de usuarios de cada red social)
X = data[['Facebook', 'Twitter', 'Instagram', 'Tik tok']]

# Crear el modelo KMeans con 3 clusters
kmeans = KMeans(n_clusters=3)

# Entrenar el modelo con los datos
kmeans.fit(X)

# Obtener las etiquetas de los clusters
labels = kmeans.labels_

# Crear la gráfica de dispersión con los clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)

# Mostrar la gráfica
plt.show()
