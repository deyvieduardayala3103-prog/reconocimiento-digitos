# Sistema interactivo de reconocimiento de dígitos

## Archivos del proyecto

| Archivo | Para qué sirve |
|---|---|
| `preprocesamiento.py` | Reglas de preprocesamiento **compartidas** por el entrenamiento y la aplicación |
| `entrenar.py` | Parte 1: división 60/20/20, dos redes, comparación, gráficas, errores, guardado |
| `verificar_modelo.py` | Recarga el modelo en un proceso nuevo y comprueba 5 muestras fijas |
| `verificar_preprocesamiento.py` | Demuestra que la app y el entrenamiento usan el mismo preprocesamiento |
| `app.py` | Parte 2: aplicación Tkinter (lienzo, 8x8, top 3, gráfico, aviso, historial, progreso) |
| `resumen_dibujos.py` | Parte 3: accuracy y avisos de los 20 dibujos propios |
| `pruebas_ruido.py` | Parte 3: 30 imágenes de prueba con ruido gaussiano 0.15 |
| `informe_final.py` | Paso 8: tabla y gráfico comparando prueba, dibujos propios y ruido |

## Orden de ejecución (en la terminal, dentro de la carpeta)

```powershell
python entrenar.py                    # 1. entrena y guarda el modelo
python verificar_modelo.py            # 2. persistencia del modelo (proceso nuevo)
python verificar_preprocesamiento.py  # 3. coherencia app / entrenamiento
python app.py                         # 4. aplicación: hacer los 20 dibujos
python resumen_dibujos.py             # 5. resultado de los 20 dibujos
python pruebas_ruido.py               # 6. ruido (al final, sin reentrenar)
python informe_final.py               # 7. diagnóstico comparativo
```

## Prueba de los 20 dibujos propios (dos por dígito)

1. Antes de empezar, borre `historial.csv` y la carpeta `dibujos` si hizo pruebas de práctica
   (o deje la casilla de prueba desmarcada en las prácticas).
2. En `app.py` marque **"Es un dibujo de la prueba oficial (primer intento)"**.
3. Dibuje → **Reconocer** (o Ctrl+Enter) → escriba el dígito → **Registrar** (o Enter) → **Limpiar** (o Esc).
4. El panel **Prueba oficial** muestra cuántos lleva de cada dígito (x/2) y la accuracy.
5. **No repita un dibujo si sale mal**: se conserva el primer intento.
6. Cada dibujo registrado se guarda en la carpeta `dibujos` (lienzo y 8x8) como evidencia.

## Demostración del flujo completo

Iniciar `app.py` → dibujar → Reconocer → Registrar → cerrar la ventana →
volver a ejecutar `app.py` → la barra inferior dice "Modelo cargado sin entrenar" y
la tabla muestra los registros anteriores → dibujar y reconocer otra vez.

Modelo ausente: cambie temporalmente el nombre de la carpeta `modelo` y abra la app;
aparece un mensaje claro y el botón Reconocer se desactiva.
