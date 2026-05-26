import numpy as np
import matplotlib.pyplot as plt

# Límites de búsqueda para la función Sphere (comúnmente [-5.12, 5.12])
BOUND_MIN = -5.12
BOUND_MAX = 5.12

# Generar la cuadrícula de datos (X, Y)
x = np.linspace(BOUND_MIN, BOUND_MAX, 100)
y = np.linspace(BOUND_MIN, BOUND_MAX, 100)
X, Y = np.meshgrid(x, y)

# Función matemática Sphere en 2 variables: f(x,y) = x^2 + y^2
Z = X**2 + Y**2

# Configurar la figura en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dibujar la superficie usando el colormap 'magma' como en test_sphere_usade.py
surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')

# Añadir barra de colores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Valor de f(x,y)')

# Añadir etiquetas y título
ax.set_title('Función Sphere en 3D')
ax.set_xlabel('Dimensión 1 (X)')
ax.set_ylabel('Dimensión 2 (Y)')
ax.set_zlabel('f(X, Y)')

# Guardar la imagen
output_path = 'sphere_3d_magma.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Imagen generada y guardada exitosamente como '{output_path}'")
