"""
preprocesamiento.py
-------------------
Reglas de preprocesamiento COMPARTIDAS por el entrenamiento y la aplicación.
Ambos scripts importan este mismo archivo, así se garantiza que la red
recibe siempre los datos con la misma escala y el mismo formato.

Cómo se construyó load_digits (documentación de scikit-learn):
  - Cada dígito se escaneó como un mapa de bits de 32 x 32 (tinta = 1, fondo = 0).
  - Ese mapa se dividió en bloques de 4 x 4 y se contaron los píxeles con tinta.
  - Resultado: una imagen de 8 x 8 con valores de 0 a 16 (16 = bloque lleno de tinta).

La aplicación imita ese mismo proceso con el dibujo del lienzo.
"""

import numpy as np
from PIL import Image

ESCALA_ORIGINAL = 16.0      # valor máximo de un píxel en load_digits
TAM_BITMAP = 32             # tamaño del mapa de bits intermedio (32 x 32)
TAM_BLOQUE = 4              # 32 / 4 = 8 -> imagen final de 8 x 8
TAM_DIGITO = 32             # alto del dígito dentro del mapa de 32 x 32
UMBRAL_TINTA = 128          # 0-255: a partir de aquí un píxel cuenta como trazo
# Medido en el conjunto de ENTRENAMIENTO: el 100 % de los dígitos de load_digits
# ocupa las 8 filas (todo el alto) y entre 4 y 7 columnas, centrados por su masa.

DESCRIPCION = {
    "entrada": "vector de 64 valores (imagen 8x8 aplanada por filas)",
    "escala_original": "0 a 16 (load_digits)",
    "normalizacion": "valor / 16  -> rango 0 a 1",
    "fondo": "0 (blanco en el lienzo se invierte a 0)",
    "tinta": "valores altos (trazo negro en el lienzo se invierte a valores altos)",
    "transformacion_lienzo": [
        "1. Inversión de colores: lienzo blanco con trazo negro -> fondo 0, tinta alta (como load_digits)",
        "2. Recorte al rectángulo que contiene el trazo (elimina espacio vacío)",
        f"3. Redimensionar conservando proporción para que el alto mida {TAM_DIGITO} px "
        "(en load_digits todos los dígitos ocupan el alto completo)",
        f"4. Centrar horizontalmente dentro de un mapa de {TAM_BITMAP}x{TAM_BITMAP} usando el centro de masa",
        f"5. Proporción de tinta en cada bloque de {TAM_BLOQUE}x{TAM_BLOQUE}",
        "6. Multiplicar por 16 -> 8x8 con valores 0 a 16 (bloque lleno = 16)",
        "7. Dividir entre 16 -> 0 a 1 (la misma normalización del entrenamiento)",
    ],
}


def normalizar(X):
    """Escala de 0-16 a 0-1. La usan el entrenamiento y la aplicación."""
    return np.asarray(X, dtype="float32") / ESCALA_ORIGINAL


def lienzo_vacio(imagen_pil):
    """True si el lienzo no tiene ningún trazo."""
    gris = np.array(imagen_pil.convert("L"))
    return not np.any(gris < UMBRAL_TINTA)


def imagen_a_8x8(imagen_pil):
    """
    Convierte el dibujo del lienzo (fondo blanco, trazo negro) en una
    imagen 8x8 con valores de 0 a 16, igual que load_digits.
    Devuelve None si el lienzo está vacío.
    """
    gris = np.array(imagen_pil.convert("L"), dtype=np.float32)

    # 1. Inversión: el lienzo es blanco (255) con trazo negro (0);
    #    load_digits tiene fondo 0 y tinta alta.
    tinta = 255.0 - gris

    # 2. Recorte al rectángulo que contiene el trazo
    filas = np.where(tinta.max(axis=1) > UMBRAL_TINTA)[0]
    cols = np.where(tinta.max(axis=0) > UMBRAL_TINTA)[0]
    if len(filas) == 0 or len(cols) == 0:
        return None
    recorte = tinta[filas[0]:filas[-1] + 1, cols[0]:cols[-1] + 1]

    # 3. Redimensionar conservando proporción (un "1" sigue siendo delgado).
    #    El alto pasa a 32 px; si el dibujo es muy ancho, manda el ancho.
    alto, ancho = recorte.shape
    factor = min(TAM_DIGITO / alto, TAM_BITMAP / ancho)
    nuevo_alto = max(1, int(round(alto * factor)))
    nuevo_ancho = max(1, int(round(ancho * factor)))
    img = Image.fromarray(recorte.astype(np.uint8))
    img = img.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
    pequeno = np.array(img, dtype=np.float32)

    # 4. Centrar en un mapa de 32x32 usando el centro de masa del trazo
    mapa = np.zeros((TAM_BITMAP, TAM_BITMAP), dtype=np.float32)
    total = pequeno.sum()
    if total > 0:
        yy, xx = np.indices(pequeno.shape)
        cy = (yy * pequeno).sum() / total
        cx = (xx * pequeno).sum() / total
    else:
        cy, cx = nuevo_alto / 2, nuevo_ancho / 2
    y0 = int(round(TAM_BITMAP / 2 - cy))
    x0 = int(round(TAM_BITMAP / 2 - cx))
    # Mantener el dígito completo dentro del mapa
    y0 = min(max(y0, 0), TAM_BITMAP - nuevo_alto)
    x0 = min(max(x0, 0), TAM_BITMAP - nuevo_ancho)
    mapa[y0:y0 + nuevo_alto, x0:x0 + nuevo_ancho] = pequeno

    # 5-6. Proporción de tinta en cada bloque de 4x4, escalada a 0-16.
    #      En un trazo negro puro equivale a contar píxeles con tinta (como load_digits).
    proporcion = (mapa / 255.0).reshape(8, TAM_BLOQUE, 8, TAM_BLOQUE).mean(axis=(1, 3))
    bloques = np.clip(np.round(proporcion * ESCALA_ORIGINAL), 0, ESCALA_ORIGINAL)
    return bloques


def imagen_a_vector(imagen_pil):
    """
    Dibujo del lienzo -> (vector de 64 valores en escala 0-1, imagen 8x8 en 0-16).
    Devuelve (None, None) si el lienzo está vacío.
    """
    img8 = imagen_a_8x8(imagen_pil)
    if img8 is None:
        return None, None
    vector = normalizar(img8.reshape(1, 64))
    return vector, img8
