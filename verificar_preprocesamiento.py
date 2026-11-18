"""
verificar_preprocesamiento.py  -  ¿La app y el entrenamiento usan el mismo preprocesamiento?
--------------------------------------------------------------------------------------------
Comprobaciones:
  1. La app y el entrenamiento importan la MISMA función normalizar() del
     archivo preprocesamiento.py (no hay dos copias del código).
  2. normalizar() deja los datos de load_digits en el rango 0-1.
  3. Se toman imágenes de PRUEBA de load_digits, se "dibujan" en un lienzo de
     280x280 (fondo blanco, trazo negro, como el de la app) y se pasan por la
     MISMA cadena de la app (imagen_a_vector). Se comparan las predicciones.

Ejecutar:  python verificar_preprocesamiento.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_digits
from tensorflow import keras

import preprocesamiento
from preprocesamiento import normalizar, imagen_a_vector


def importa_de_preprocesamiento(archivo):
    """Revisa en el código fuente qué funciones importa un script desde preprocesamiento.py."""
    arbol = ast.parse(open(archivo, encoding="utf-8").read())
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module == "preprocesamiento":
            nombres.update(a.name for a in nodo.names)
    return nombres


print("1. ORIGEN DEL PREPROCESAMIENTO")
print("   entrenar.py importa:", sorted(importa_de_preprocesamiento("entrenar.py")))
print("   app.py importa:     ", sorted(importa_de_preprocesamiento("app.py")))
print("   imagen_a_vector() usa internamente normalizar():",
      "normalizar(" in open("preprocesamiento.py", encoding="utf-8").read().split("def imagen_a_vector")[1])

digitos = load_digits()
X = normalizar(digitos.data)
print("\n2. ESCALA")
print(f"   Original: {digitos.data.min():.0f} a {digitos.data.max():.0f}  ->  normalizada: {X.min():.1f} a {X.max():.1f}"
      f"  (división entre {preprocesamiento.ESCALA_ORIGINAL:.0f})")

# 3. Pasar imágenes de prueba por la cadena de la aplicación
modelo = keras.models.load_model(os.path.join("modelo", "modelo_final.keras"))
idx_prueba = np.load(os.path.join("modelo", "indices_division.npz"))["prueba"]


def como_lienzo(img8):
    """Convierte una imagen 8x8 (0-16) en un 'dibujo' 280x280 fondo blanco / trazo negro."""
    gris = (255 - img8 / 16 * 255).astype(np.uint8)
    grande = Image.fromarray(gris).resize((200, 200), Image.NEAREST)
    lienzo = Image.new("L", (280, 280), 255)
    lienzo.paste(grande, (40, 40))
    return lienzo


vectores_app = []
for i in idx_prueba:
    v, _ = imagen_a_vector(como_lienzo(digitos.images[i]))
    vectores_app.append(v[0])
vectores_app = np.array(vectores_app, dtype="float32")

y = digitos.target[idx_prueba]
pred_directa = modelo.predict(X[idx_prueba], verbose=0).argmax(axis=1)
pred_app = modelo.predict(vectores_app, verbose=0).argmax(axis=1)

print("\n3. CADENA DE LA APLICACIÓN SOBRE IMÁGENES DE PRUEBA")
print(f"   Imágenes: {len(y)}")
print(f"   Rango de los vectores de la app: {vectores_app.min():.1f} a {vectores_app.max():.1f}")
print(f"   Accuracy entrada directa (entrenamiento): {(pred_directa == y).mean():.2%}")
print(f"   Accuracy pasando por la cadena de la app:  {(pred_app == y).mean():.2%}")
print(f"   Predicciones iguales en ambos caminos:     {(pred_directa == pred_app).mean():.2%}")
print("   (No es idéntica al 100 %: recortar y centrar por masa desplaza un poco los bloques.)")

# Imagen comparativa
fig, ejes = plt.subplots(3, 6, figsize=(10, 5.4))
for j in range(6):
    i = idx_prueba[j]
    ejes[0, j].imshow(digitos.images[i], cmap="gray_r", vmin=0, vmax=16)
    ejes[0, j].set_title(f"load_digits ({y[j]})", fontsize=8)
    ejes[1, j].imshow(como_lienzo(digitos.images[i]), cmap="gray", vmin=0, vmax=255)
    ejes[1, j].set_title("como lienzo", fontsize=8)
    ejes[2, j].imshow(vectores_app[j].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ejes[2, j].set_title(f"cadena app -> {pred_app[j]}", fontsize=8)
    for k in range(3):
        ejes[k, j].axis("off")
fig.suptitle("Misma imagen por el camino del entrenamiento y por el de la aplicación")
fig.tight_layout()
fig.savefig("verificacion_preprocesamiento.png", dpi=150)
print("\nGuardado: verificacion_preprocesamiento.png")
