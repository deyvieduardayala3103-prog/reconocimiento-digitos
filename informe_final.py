"""
informe_final.py  -  Diagnóstico (paso 8)
------------------------------------------
Reúne en una sola tabla y un gráfico el desempeño en:
  - conjunto de prueba de load_digits  (resultados_entrenamiento.json)
  - 20 dibujos propios                  (resultados_dibujos.json)
  - 30 imágenes de prueba con ruido     (resultados_ruido.json)

Ejecutar AL FINAL, después de:
  python entrenar.py
  python resumen_dibujos.py
  python pruebas_ruido.py
"""

import json
import os
import matplotlib.pyplot as plt


def leer(ruta):
    if not os.path.exists(ruta):
        print(f"  Falta {ruta} (ejecute primero el script que lo genera)")
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


entrenamiento = leer(os.path.join("modelo", "resultados_entrenamiento.json"))
dibujos = leer("resultados_dibujos.json")
ruido = leer("resultados_ruido.json")

filas = []
if entrenamiento:
    elegido = entrenamiento["modelo_elegido"]
    r = entrenamiento["resultados"][elegido]
    filas.append(("Prueba\n(load_digits)", r["prueba"]["accuracy"],
                  f"{len(entrenamiento['resultados'])} modelos; elegido: {elegido}; "
                  f"F1 macro {r['prueba']['f1_macro']:.4f}"))
if ruido:
    filas.append(("30 imágenes\noriginales", ruido["accuracy_original"],
                  f"semilla {ruido['semilla']}"))
    filas.append(("30 imágenes\ncon ruido", ruido["accuracy_ruido"],
                  f"desviación {ruido['desviacion']}; cambios: {len(ruido['cambios'])}"))
if dibujos:
    filas.append(("Dibujos\npropios", dibujos["accuracy"],
                  f"{dibujos['correctos']}/{dibujos['total']} correctos; "
                  f"avisos baja confianza: {dibujos['avisos_baja_confianza']}"))

print("\nDIAGNÓSTICO: COMPARACIÓN DE DESEMPEÑO")
print(f"{'Conjunto':<24}{'Accuracy':>10}   Detalle")
for nombre, acc, detalle in filas:
    print(f"{nombre.replace(chr(10), ' '):<24}{acc:>10.2%}   {detalle}")

if filas:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    nombres = [f[0] for f in filas]
    valores = [f[1] * 100 for f in filas]
    colores = ["#1f6fe5", "#2ca02c", "#d62728", "#e07b00"][:len(filas)]
    barras = ax.bar(nombres, valores, color=colores, edgecolor="black")
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f} %",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparación del desempeño del modelo congelado")
    fig.tight_layout()
    fig.savefig("comparacion_desempeno.png", dpi=150)
    print("\nGuardado: comparacion_desempeno.png")

print("""
POR QUÉ PUEDEN DIFERIR (para el informe):
- Prueba load_digits: mismas condiciones que el entrenamiento (mismos escritores,
  mismo escáner, mismo proceso de 32x32 a 8x8).
- Dibujos propios: otro "escritor", trazo con ratón, grosor y estilo distintos;
  el preprocesamiento intenta acercarlos, pero la distribución cambia.
- Ruido: agrega "tinta" falsa en el fondo; los dígitos de trazo fino (1, 7)
  pierden contraste frente al ruido.

MEJORA CONCRETA PARA LA SIGUIENTE VERSIÓN:
- Entrenar con aumento de datos (pequeños desplazamientos, rotaciones, grosor y
  ruido gaussiano) y agregar al entrenamiento dibujos hechos con el lienzo,
  evaluando después con un conjunto de prueba NUEVO (sin tocar el actual).
""")

if filas:
    print("Cierre la ventana del gráfico para terminar.")
    plt.show()
