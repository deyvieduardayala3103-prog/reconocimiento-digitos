"""
resumen_dibujos.py  -  Resultado de los 20 dibujos propios
-----------------------------------------------------------
Lee historial.csv y resume SOLO los registros marcados como
"prueba" en la aplicación (primer intento de cada dibujo, sin borrar errores).

Ejecutar:  python resumen_dibujos.py
"""

import csv
import json
from collections import Counter

ARCHIVO = "historial.csv"

with open(ARCHIVO, newline="", encoding="utf-8") as f:
    filas = [fila for fila in csv.DictReader(f) if fila["tipo"] == "prueba"]

if not filas:
    print("No hay registros de tipo 'prueba' en", ARCHIVO)
    raise SystemExit

total = len(filas)
correctos = sum(1 for f in filas if f["resultado"] == "correcto")
avisos = sum(1 for f in filas if f["baja_confianza"] == "si")
por_digito = Counter(f["etiqueta"] for f in filas)

print("RESUMEN DE DIBUJOS PROPIOS")
print(f"Dibujos registrados: {total} (meta: 20, dos por dígito)")
print("Dibujos por dígito:", dict(sorted(por_digito.items())))
print(f"Correctos: {correctos} | Incorrectos: {total - correctos}")
print(f"Accuracy dibujos propios: {correctos / total:.2%}")
print(f"Avisos de baja confianza (< 0.70): {avisos}")

print("\nErrores:")
for f in filas:
    if f["resultado"] != "correcto":
        print(f"  {f['fecha_hora']} | quiso {f['etiqueta']} -> predijo {f['prediccion']} "
              f"(p={float(f['prob_maxima']):.2f})")

with open("resultados_dibujos.json", "w", encoding="utf-8") as f:
    json.dump({"total": total, "correctos": correctos,
               "accuracy": correctos / total, "avisos_baja_confianza": avisos},
              f, ensure_ascii=False, indent=2)
print("\nGuardado: resultados_dibujos.json")
