"""
pruebas_ruido.py  -  Resistencia al ruido (se ejecuta AL FINAL)
----------------------------------------------------------------
Toma 30 imágenes del conjunto de PRUEBA elegidas con semilla fija,
les añade ruido gaussiano (desviación 0.15 en escala 0-1), recorta a [0, 1]
y compara la exactitud original contra la exactitud con ruido.
El modelo NO se reentrena ni se ajusta con estos resultados.

Ejecutar:  python pruebas_ruido.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from tensorflow import keras

from preprocesamiento import normalizar

SEMILLA_RUIDO = 2026
DESVIACION = 0.15
CANTIDAD = 30

modelo = keras.models.load_model(os.path.join("modelo", "modelo_final.keras"))
idx_prueba = np.load(os.path.join("modelo", "indices_division.npz"))["prueba"]

digitos = load_digits()
generador = np.random.default_rng(SEMILLA_RUIDO)
elegidos = np.sort(generador.choice(idx_prueba, size=CANTIDAD, replace=False))

X = normalizar(digitos.data[elegidos])
y = digitos.target[elegidos]

X_ruido = np.clip(X + generador.normal(0, DESVIACION, X.shape), 0.0, 1.0).astype("float32")

pred_original = modelo.predict(X, verbose=0).argmax(axis=1)
pred_ruido = modelo.predict(X_ruido, verbose=0).argmax(axis=1)

acc_original = float((pred_original == y).mean())
acc_ruido = float((pred_ruido == y).mean())

print("PRUEBA DE RUIDO GAUSSIANO")
print(f"Imágenes: {CANTIDAD} del conjunto de prueba | semilla {SEMILLA_RUIDO} | desviación {DESVIACION}")
print(f"Exactitud original:  {acc_original:.2%}")
print(f"Exactitud con ruido: {acc_ruido:.2%}")

conf_original = modelo.predict(X, verbose=0).max(axis=1).mean()
conf_ruido = modelo.predict(X_ruido, verbose=0).max(axis=1).mean()
print(f"Confianza promedio original:  {conf_original:.3f}")
print(f"Confianza promedio con ruido: {conf_ruido:.3f}")

cambios = np.where(pred_original != pred_ruido)[0]
errores_ruido = []
print(f"\nPredicciones que cambiaron por el ruido: {len(cambios)}")
for i in cambios:
    print(f"  Índice {elegidos[i]}: real {y[i]} | original {pred_original[i]} -> con ruido {pred_ruido[i]}")
    errores_ruido.append({"indice": int(elegidos[i]), "real": int(y[i]),
                          "original": int(pred_original[i]), "con_ruido": int(pred_ruido[i])})

# Imagen de ejemplo: original vs con ruido (primero las que cambiaron)
n = min(6, CANTIDAD)
otras = [i for i in range(CANTIDAD) if i not in set(cambios)]
mostrar = (list(cambios) + otras)[:n]
fig, ejes = plt.subplots(2, n, figsize=(2.2 * n, 6.4))
for j, i in enumerate(mostrar):
    color_o = "black" if pred_original[i] == y[i] else "red"
    color_r = "black" if pred_ruido[i] == y[i] else "red"
    ejes[0, j].imshow(X[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ejes[0, j].set_title(f"Real {y[i]}\nPred {pred_original[i]}", fontsize=10, color=color_o)
    ejes[1, j].imshow(X_ruido[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ejes[1, j].set_title(f"Con ruido\nPred {pred_ruido[i]}", fontsize=10, color=color_r)
    ejes[0, j].axis("off")
    ejes[1, j].axis("off")
fig.suptitle(f"Original (arriba) vs ruido gaussiano σ={DESVIACION} (abajo) - errores en rojo")
fig.tight_layout()
fig.savefig("ejemplos_ruido.png", dpi=150)

with open("resultados_ruido.json", "w", encoding="utf-8") as f:
    json.dump({
        "semilla": SEMILLA_RUIDO, "desviacion": DESVIACION, "cantidad": CANTIDAD,
        "indices": elegidos.tolist(),
        "accuracy_original": acc_original, "accuracy_ruido": acc_ruido,
        "confianza_original": float(conf_original), "confianza_ruido": float(conf_ruido),
        "cambios": errores_ruido,
    }, f, ensure_ascii=False, indent=2)

print("\nGuardado: resultados_ruido.json y ejemplos_ruido.png")
print("Cierre la ventana del gráfico para terminar.")
plt.show()
