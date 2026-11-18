"""
entrenar.py  -  PRIMERA PARTE: entrenamiento y evaluación
-----------------------------------------------------------
1. Preparación reproducible (división 60/20/20, semilla, estratificación, índices)
2. Dos modelos (Sequential y API funcional) con EarlyStopping
3. Comparación en validación, elección del modelo final y evaluación en prueba
4. Curvas de pérdida, matriz de confusión y análisis de errores
5. Guardado del modelo, del preprocesamiento y de 5 muestras de referencia

Ejecutar:  python entrenar.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # solo guarda imágenes, no abre ventanas
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from preprocesamiento import normalizar, DESCRIPCION

SEMILLA = 42
CARPETA = "modelo"
os.makedirs(CARPETA, exist_ok=True)
keras.utils.set_random_seed(SEMILLA)

# ======================================================================
# 1. PREPARACIÓN REPRODUCIBLE
# ======================================================================
digitos = load_digits()
X_original = digitos.data          # (1797, 64) valores 0 a 16
y = digitos.target                 # (1797,) dígitos 0 a 9
indices = np.arange(len(y))

print("DATOS load_digits")
print("Imágenes:", X_original.shape, "| Rango original:", X_original.min(), "a", X_original.max())

# 60 % entrenamiento, 20 % validación, 20 % prueba (estratificado)
idx_entrena, idx_temp = train_test_split(
    indices, test_size=0.40, random_state=SEMILLA, stratify=y)
idx_valida, idx_prueba = train_test_split(
    idx_temp, test_size=0.50, random_state=SEMILLA, stratify=y[idx_temp])

# Conservar los índices de la división
np.savez(os.path.join(CARPETA, "indices_division.npz"),
         entrena=idx_entrena, valida=idx_valida, prueba=idx_prueba)

# Normalización 0-16 -> 0-1 (la misma función que usa la aplicación)
X = normalizar(X_original)
X_entrena, y_entrena = X[idx_entrena], y[idx_entrena]
X_valida, y_valida = X[idx_valida], y[idx_valida]
X_prueba, y_prueba = X[idx_prueba], y[idx_prueba]

print(f"Entrenamiento: {len(idx_entrena)} | Validación: {len(idx_valida)} | Prueba: {len(idx_prueba)}")
print("Rango normalizado:", X.min(), "a", X.max())

# ======================================================================
# 2. DOS MODELOS
# ======================================================================
CONFIG = {
    "semilla": SEMILLA,
    "division": "60/20/20 estratificada",
    "epocas_maximas": 300,
    "batch_size": 32,
    "early_stopping": {"monitor": "val_loss", "patience": 20, "restore_best_weights": True},
    "secuencial": {
        "capas_ocultas": [64, 32],
        "activacion": "relu",
        "regularizacion": "Dropout 0.20 tras la primera capa oculta",
        "tasa_aprendizaje": 0.001,
        "justificacion": "Dos capas medianas: suficiente capacidad para 64 entradas; "
                         "Dropout reduce el sobreajuste con solo ~1078 ejemplos.",
    },
    "funcional": {
        "capas_ocultas": [128],
        "activacion": "relu",
        "regularizacion": "L2 0.0001 + Dropout 0.30",
        "tasa_aprendizaje": 0.001,
        "justificacion": "Una capa más ancha y poco profunda; L2 y Dropout controlan "
                         "el sobreajuste de una capa grande.",
    },
}


def crear_secuencial(cfg):
    modelo = keras.Sequential([
        keras.Input(shape=(64,)),
        layers.Dense(cfg["capas_ocultas"][0], activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(cfg["capas_ocultas"][1], activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="red_secuencial")
    return modelo


def crear_funcional(cfg):
    entrada = keras.Input(shape=(64,), name="pixeles_8x8")
    x = layers.Dense(cfg["capas_ocultas"][0], activation="relu",
                     kernel_regularizer=regularizers.l2(0.0001))(entrada)
    x = layers.Dropout(0.30)(x)
    salida = layers.Dense(10, activation="softmax", name="probabilidades")(x)
    return keras.Model(inputs=entrada, outputs=salida, name="red_funcional")


def entrenar(modelo, cfg_modelo):
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg_modelo["tasa_aprendizaje"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    parada = keras.callbacks.EarlyStopping(**CONFIG["early_stopping"])
    historial = modelo.fit(
        X_entrena, y_entrena,
        validation_data=(X_valida, y_valida),
        epochs=CONFIG["epocas_maximas"],
        batch_size=CONFIG["batch_size"],
        callbacks=[parada],
        verbose=0,
    )
    # Épocas efectivas = época con mejor val_loss (pesos restaurados)
    epocas_totales = len(historial.history["loss"])
    epoca_mejor = int(np.argmin(historial.history["val_loss"])) + 1
    return historial, epocas_totales, epoca_mejor


def metricas(modelo, X_, y_):
    pred = np.argmax(modelo.predict(X_, verbose=0), axis=1)
    return {
        "accuracy": float(accuracy_score(y_, pred)),
        "f1_macro": float(f1_score(y_, pred, average="macro")),
    }, pred


resultados = {}
modelos = {}
historiales = {}

for nombre, crear in [("secuencial", crear_secuencial), ("funcional", crear_funcional)]:
    print(f"\nEntrenando modelo {nombre}...")
    modelo = crear(CONFIG[nombre])
    historial, ep_tot, ep_mejor = entrenar(modelo, CONFIG[nombre])
    m_val, _ = metricas(modelo, X_valida, y_valida)
    resultados[nombre] = {
        "parametros": int(modelo.count_params()),
        "epocas_ejecutadas": ep_tot,
        "epocas_efectivas": ep_mejor,
        "validacion": m_val,
    }
    modelos[nombre] = modelo
    historiales[nombre] = historial
    print(f"  Validación -> accuracy {m_val['accuracy']:.4f} | F1 macro {m_val['f1_macro']:.4f}")

# ======================================================================
# 3. ELECCIÓN POR VALIDACIÓN Y EVALUACIÓN FINAL EN PRUEBA
# ======================================================================
# Criterio: mayor F1 macro en validación; empate -> menos parámetros.
elegido = max(resultados, key=lambda n: (resultados[n]["validacion"]["f1_macro"],
                                         -resultados[n]["parametros"]))

# Los dos diseños ya están cerrados: ahora sí se evalúan en prueba
predicciones_prueba = {}
for nombre, modelo in modelos.items():
    m_prueba, pred = metricas(modelo, X_prueba, y_prueba)
    resultados[nombre]["prueba"] = m_prueba
    predicciones_prueba[nombre] = pred

print("\nCOMPARACIÓN DE MODELOS")
print(f"{'Modelo':<12}{'Parámetros':>11}{'Épocas ef.':>11}{'Val acc':>9}{'Val F1':>8}{'Test acc':>10}{'Test F1':>9}")
for n, r in resultados.items():
    print(f"{n:<12}{r['parametros']:>11}{r['epocas_efectivas']:>11}"
          f"{r['validacion']['accuracy']:>9.4f}{r['validacion']['f1_macro']:>8.4f}"
          f"{r['prueba']['accuracy']:>10.4f}{r['prueba']['f1_macro']:>9.4f}")
print(f"\nModelo elegido (por F1 macro de validación): {elegido}")

modelo_final = modelos[elegido]
pred_final = predicciones_prueba[elegido]

# ======================================================================
# 4. CURVAS, MATRIZ DE CONFUSIÓN Y ANÁLISIS DE ERRORES
# ======================================================================
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (nombre, h) in zip(ejes, historiales.items()):
    ax.plot(h.history["loss"], label="Entrenamiento")
    ax.plot(h.history["val_loss"], label="Validación")
    ax.axvline(resultados[nombre]["epocas_efectivas"] - 1, color="gray",
               linestyle="--", label="Mejor época")
    ax.set_title(f"Pérdida - modelo {nombre}")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
fig.tight_layout()
fig.savefig("curvas_perdida.png", dpi=150)
plt.close(fig)

matriz = confusion_matrix(y_prueba, pred_final, labels=list(range(10)))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(matriz, display_labels=list(range(10))).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusión en prueba - modelo {elegido}")
ax.set_xlabel("Dígito predicho")
ax.set_ylabel("Dígito real")
fig.tight_layout()
fig.savefig("matriz_confusion.png", dpi=150)
plt.close(fig)

print("\nMATRIZ DE CONFUSIÓN (prueba)")
print(matriz)

# Análisis de tres errores reales (o todos si hay menos)
probs_prueba = modelo_final.predict(X_prueba, verbose=0)
errores = np.where(pred_final != y_prueba)[0]
analizados = errores[:3]
print(f"\nERRORES EN PRUEBA: {len(errores)} de {len(y_prueba)}")
detalle_errores = []
if len(analizados) > 0:
    fig, ejes = plt.subplots(1, len(analizados), figsize=(3.2 * len(analizados), 3.6))
    ejes = np.atleast_1d(ejes)
    for ax, i in zip(ejes, analizados):
        idx_global = int(idx_prueba[i])
        top2 = np.argsort(probs_prueba[i])[::-1][:2]
        info = {
            "indice_dataset": idx_global,
            "real": int(y_prueba[i]),
            "predicho": int(pred_final[i]),
            "prob_predicho": float(probs_prueba[i][pred_final[i]]),
            "prob_real": float(probs_prueba[i][y_prueba[i]]),
            "segunda_opcion": int(top2[1]),
        }
        detalle_errores.append(info)
        print(f"  Índice {idx_global}: real {info['real']} -> predicho {info['predicho']} "
              f"(p={info['prob_predicho']:.2f}; prob. del real={info['prob_real']:.2f})")
        ax.imshow(X_original[idx_global].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"Real {info['real']} / Pred {info['predicho']}\np={info['prob_predicho']:.2f}")
        ax.axis("off")
    fig.suptitle("Errores reales del modelo final")
    fig.tight_layout()
    fig.savefig("errores_analizados.png", dpi=150)
    plt.close(fig)

# ======================================================================
# 5. PERSISTENCIA
# ======================================================================
modelo_final.save(os.path.join(CARPETA, "modelo_final.keras"))

with open(os.path.join(CARPETA, "preprocesamiento.json"), "w", encoding="utf-8") as f:
    json.dump(DESCRIPCION, f, ensure_ascii=False, indent=2)

# Cinco muestras fijas de prueba (las 5 primeras del conjunto de prueba)
muestras_idx = idx_prueba[:5]
probs_ref = modelo_final.predict(X[muestras_idx], verbose=0)
np.savez(os.path.join(CARPETA, "muestras_referencia.npz"),
         indices=muestras_idx, probabilidades=probs_ref)

with open(os.path.join(CARPETA, "resultados_entrenamiento.json"), "w", encoding="utf-8") as f:
    json.dump({
        "configuracion": CONFIG,
        "resultados": resultados,
        "modelo_elegido": elegido,
        "criterio_eleccion": "Mayor F1 macro en validación (empate: menos parámetros)",
        "errores_prueba": len(errores),
        "errores_analizados": detalle_errores,
    }, f, ensure_ascii=False, indent=2)

print(f"\nGuardado en la carpeta '{CARPETA}':")
print("  modelo_final.keras, preprocesamiento.json, indices_division.npz,")
print("  muestras_referencia.npz, resultados_entrenamiento.json")
print("Imágenes: curvas_perdida.png, matriz_confusion.png, errores_analizados.png")
