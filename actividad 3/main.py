import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar el dataset
data = pd.read_csv('dataset.csv')
import matplotlib.pyplot as plt

# Datos del dataset
time = data['Year']
# Obtener nombre de las redes sociales
social_nets = data.columns.drop('Year')

# Get data and print them
for network in social_nets:
  plt.scatter(time, data[network], label=network)

# Configuraciones de la gráfica
plt.title('Número de usuarios de redes sociales por año')
plt.xlabel('Año')
plt.ylabel('Número de usuarios')
plt.legend()

# Mostrar la gráfica
plt.show()


# -------------

# Seleccionar la variable independiente (años)
x_axis = data[['Year']]

# Seleccionar las variables dependientes (cantidad de usuarios de cada red social)
y_axis = data[['Facebook', 'Twitter', 'Instagram', 'Tik tok']]

# Crear el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
model.fit(x_axis, y_axis)

# Predecir la tendencia futura de usuarios de redes sociales en los próximos 4 años
future_years = pd.DataFrame({'Year': [2024, 2025, 2026, 2027]})
future_users = model.predict(future_years)

# Cálculo del coeficiente de determinación (R²)
r2 = r2_score(y_axis, model.predict(x_axis))

# Valor de R al cuadrado
print(f"R al cuadrado: {r2}")
print(' ')
# Imprimir los resultados de la predicción
print(future_users)
