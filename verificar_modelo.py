"""
verificar_modelo.py  -  Persistencia del modelo
------------------------------------------------
Se ejecuta en un PROCESO NUEVO (después de entrenar.py).
Recarga el modelo guardado y comprueba que conserva sus predicciones
sobre 5 muestras fijas, dentro de una tolerancia numérica declarada.

Ejecutar:  python verificar_modelo.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.datasets import load_digits
from tensorflow import keras

from preprocesamiento import normalizar

TOLERANCIA = 1e-5   # diferencia máxima permitida entre probabilidades

modelo = keras.models.load_model(os.path.join("modelo", "modelo_final.keras"))
referencia = np.load(os.path.join("modelo", "muestras_referencia.npz"))

digitos = load_digits()
indices = referencia["indices"]
X = normalizar(digitos.data[indices])

probs_nuevas = modelo.predict(X, verbose=0)
probs_guardadas = referencia["probabilidades"]

print("VERIFICACIÓN DEL MODELO RECARGADO (proceso nuevo)")
print(f"Tolerancia declarada: {TOLERANCIA}")
print(f"{'Índice':>7}{'Real':>6}{'Pred. guardada':>16}{'Pred. recargada':>17}{'Dif. máxima':>14}")
for i, idx in enumerate(indices):
    dif = np.abs(probs_nuevas[i] - probs_guardadas[i]).max()
    print(f"{idx:>7}{digitos.target[idx]:>6}{probs_guardadas[i].argmax():>16}"
          f"{probs_nuevas[i].argmax():>17}{dif:>14.2e}")

misma_clase = np.array_equal(probs_nuevas.argmax(axis=1), probs_guardadas.argmax(axis=1))
dentro_tolerancia = np.allclose(probs_nuevas, probs_guardadas, atol=TOLERANCIA)

print("\nMismas clases predichas:", "SÍ" if misma_clase else "NO")
print("Probabilidades dentro de la tolerancia:", "SÍ" if dentro_tolerancia else "NO")
print("RESULTADO:", "VERIFICACIÓN CORRECTA" if (misma_clase and dentro_tolerancia) else "FALLÓ")
