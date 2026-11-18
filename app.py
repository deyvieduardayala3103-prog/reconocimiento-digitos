"""
app.py  -  SEGUNDA PARTE: aplicación interactiva (Tkinter)
------------------------------------------------------------
- Lienzo para dibujar con el ratón, botones Reconocer y Limpiar
- Muestra la imagen 8x8 que realmente recibe la red
- Dígito predicho, top 3 y gráfico de probabilidades 0-9
- Aviso de lienzo vacío y de baja confianza (< 0.70)
- Registro en historial.csv (se conserva al cerrar la aplicación)
- Contador de la prueba oficial (20 dibujos, dos por dígito) y tabla del historial
- Cada dibujo registrado se guarda como imagen en la carpeta "dibujos"
- NO entrena: solo carga el modelo guardado por entrenar.py

Ejecutar:  python app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw

from preprocesamiento import imagen_a_vector, lienzo_vacio

RUTA_MODELO = os.path.join("modelo", "modelo_final.keras")
ARCHIVO_HISTORIAL = "historial.csv"
CARPETA_DIBUJOS = "dibujos"
DIBUJOS_POR_DIGITO = 2      # prueba oficial: 2 dibujos por dígito = 20 en total
COLUMNAS = ["fecha_hora", "etiqueta", "prediccion", "prob_maxima", "resultado",
            "baja_confianza", "tipo", "vector_8x8"]
UMBRAL_CONFIANZA = 0.70     # regla didáctica fija (no implica calibración)
TAM_LIENZO = 280
GROSOR = 18


class AppDigitos:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Reconocimiento de dígitos 0-9")
        self.raiz.resizable(False, False)
        self.modelo = None
        self.ultimo = None          # resultado de la última predicción
        self.punto_anterior = None

        # Imagen PIL paralela al lienzo: es la que se procesa (fondo blanco, trazo negro)
        self.imagen = Image.new("L", (TAM_LIENZO, TAM_LIENZO), 255)
        self.dibujo = ImageDraw.Draw(self.imagen)

        self._crear_interfaz()
        self.raiz.after(100, self._cargar_modelo)

    # ------------------------------------------------------------------
    # Interfaz
    # ------------------------------------------------------------------
    def _crear_interfaz(self):
        izq = tk.Frame(self.raiz, padx=10, pady=10)
        izq.grid(row=0, column=0, sticky="n")
        der = tk.Frame(self.raiz, padx=10, pady=10)
        der.grid(row=0, column=1, sticky="n")

        # --- Lienzo de dibujo ---
        tk.Label(izq, text="Dibuje un dígito (0-9)", font=("Arial", 12, "bold")).pack()
        self.lienzo = tk.Canvas(izq, width=TAM_LIENZO, height=TAM_LIENZO, bg="white",
                                cursor="pencil", highlightthickness=2,
                                highlightbackground="#555")
        self.lienzo.pack(pady=5)
        self.lienzo.bind("<Button-1>", self._iniciar_trazo)
        self.lienzo.bind("<B1-Motion>", self._dibujar)
        self.lienzo.bind("<ButtonRelease-1>", lambda e: setattr(self, "punto_anterior", None))

        botones = tk.Frame(izq)
        botones.pack(pady=5)
        self.btn_reconocer = tk.Button(botones, text="Reconocer", width=12, bg="#2e7d32",
                                       fg="white", font=("Arial", 10, "bold"),
                                       command=self.reconocer)
        self.btn_reconocer.pack(side="left", padx=5)
        tk.Button(botones, text="Limpiar", width=12, command=self.limpiar).pack(side="left", padx=5)

        # --- Imagen 8x8 que recibe la red ---
        tk.Label(izq, text="Imagen 8 x 8 que recibe la red").pack(pady=(10, 0))
        self.vista8 = tk.Canvas(izq, width=160, height=160, bg="white",
                                highlightthickness=1, highlightbackground="#999")
        self.vista8.pack(pady=5)

        # --- Resultados ---
        tk.Label(der, text="Predicción", font=("Arial", 12, "bold")).pack()
        self.lbl_pred = tk.Label(der, text="-", font=("Arial", 48, "bold"), fg="#1565c0")
        self.lbl_pred.pack()
        self.lbl_top3 = tk.Label(der, text="Top 3: -", font=("Arial", 11), justify="left")
        self.lbl_top3.pack()
        self.lbl_aviso = tk.Label(der, text="", font=("Arial", 10, "bold"), fg="#c62828",
                                  wraplength=360)
        self.lbl_aviso.pack(pady=3)

        tk.Label(der, text="Probabilidades de 0 a 9").pack(pady=(8, 0))
        self.grafico = tk.Canvas(der, width=380, height=190, bg="white",
                                 highlightthickness=1, highlightbackground="#999")
        self.grafico.pack(pady=5)
        self._dibujar_barras(np.zeros(10))

        # --- Registro en historial ---
        marco = tk.LabelFrame(der, text="Registrar en historial", padx=8, pady=8)
        marco.pack(fill="x", pady=8)
        fila = tk.Frame(marco)
        fila.pack(fill="x")
        tk.Label(fila, text="Dígito que quiso dibujar (0-9):").pack(side="left")
        self.entrada_etiqueta = tk.Entry(fila, width=5, justify="center", font=("Arial", 12))
        self.entrada_etiqueta.pack(side="left", padx=5)
        self.var_prueba = tk.BooleanVar(value=False)
        tk.Checkbutton(marco, text="Es un dibujo de la prueba oficial (primer intento)",
                       variable=self.var_prueba).pack(anchor="w", pady=3)
        self.btn_registrar = tk.Button(marco, text="Registrar", width=14,
                                       command=self.registrar, state="disabled")
        self.btn_registrar.pack(pady=3)
        self.entrada_etiqueta.bind("<Return>", lambda e: self.registrar())

        # --- Progreso de la prueba oficial (20 dibujos) ---
        self.marco_prog = tk.LabelFrame(der, text="Prueba oficial: 0/20", padx=8, pady=5)
        self.marco_prog.pack(fill="x")
        fila_d = tk.Frame(self.marco_prog)
        fila_d.pack()
        self.lbl_digitos = []
        for d in range(10):
            lbl = tk.Label(fila_d, text=f"{d}\n0/2", width=4, relief="groove",
                           font=("Arial", 9))
            lbl.pack(side="left", padx=1)
            self.lbl_digitos.append(lbl)
        self.lbl_resumen = tk.Label(self.marco_prog, text="Correctos: 0 | Avisos de baja confianza: 0")
        self.lbl_resumen.pack(pady=(4, 0))

        # --- Tabla con los últimos registros del historial ---
        abajo = tk.LabelFrame(self.raiz, text="Últimos registros de historial.csv", padx=8, pady=5)
        abajo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)
        columnas = ("fecha_hora", "etiqueta", "prediccion", "prob_maxima",
                    "resultado", "baja_confianza", "tipo")
        self.tabla = ttk.Treeview(abajo, columns=columnas, show="headings", height=5)
        anchos = (150, 70, 80, 90, 90, 110, 80)
        for col, ancho in zip(columnas, anchos):
            self.tabla.heading(col, text=col)
            self.tabla.column(col, width=ancho, anchor="center")
        self.tabla.tag_configure("incorrecto", foreground="#c62828")
        self.tabla.pack(fill="x")

        self.lbl_estado = tk.Label(self.raiz, text="Cargando modelo...", anchor="w",
                                   relief="sunken", padx=5)
        self.lbl_estado.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Atajos de teclado
        self.raiz.bind("<Escape>", lambda e: self.limpiar())
        self.raiz.bind("<Control-Return>", lambda e: self.reconocer())
        self._actualizar_historial()

    # ------------------------------------------------------------------
    # Modelo
    # ------------------------------------------------------------------
    def _cargar_modelo(self):
        if not os.path.exists(RUTA_MODELO):
            self.btn_reconocer.config(state="disabled")
            self.lbl_estado.config(text="Modelo no encontrado", fg="#c62828")
            messagebox.showerror(
                "Modelo ausente",
                f"No se encontró el modelo en:\n{os.path.abspath(RUTA_MODELO)}\n\n"
                "Ejecute primero:  python entrenar.py\n"
                "La aplicación no entrena la red; solo usa el modelo guardado.")
            return
        try:
            from tensorflow import keras
            self.modelo = keras.models.load_model(RUTA_MODELO)
        except Exception as error:
            self.btn_reconocer.config(state="disabled")
            self.lbl_estado.config(text="Error al cargar el modelo", fg="#c62828")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{error}")
            return
        self.lbl_estado.config(
            text=f"Modelo cargado sin entrenar: {RUTA_MODELO} | "
                 f"Registros en historial: {self._contar_registros()}")

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    def _iniciar_trazo(self, evento):
        self.punto_anterior = (evento.x, evento.y)
        self._dibujar(evento)

    def _dibujar(self, evento):
        x, y = evento.x, evento.y
        x0, y0 = self.punto_anterior or (x, y)
        r = GROSOR / 2
        self.lienzo.create_line(x0, y0, x, y, width=GROSOR, fill="black",
                                capstyle="round", smooth=True)
        self.lienzo.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.dibujo.line([x0, y0, x, y], fill=0, width=GROSOR)
        self.dibujo.ellipse([x - r, y - r, x + r, y + r], fill=0)
        self.punto_anterior = (x, y)

    def limpiar(self):
        self.lienzo.delete("all")
        self.dibujo.rectangle([0, 0, TAM_LIENZO, TAM_LIENZO], fill=255)
        self.vista8.delete("all")
        self.lbl_pred.config(text="-")
        self.lbl_top3.config(text="Top 3: -")
        self.lbl_aviso.config(text="")
        self._dibujar_barras(np.zeros(10))
        self.ultimo = None
        self.btn_registrar.config(state="disabled")
        self.entrada_etiqueta.delete(0, "end")

    # ------------------------------------------------------------------
    # Reconocimiento
    # ------------------------------------------------------------------
    def reconocer(self):
        if self.modelo is None:
            messagebox.showerror("Modelo ausente",
                                 "No hay un modelo cargado. Ejecute primero: python entrenar.py")
            return
        if lienzo_vacio(self.imagen):
            messagebox.showwarning("Lienzo vacío", "Dibuje un dígito antes de reconocer.")
            return

        vector, img8 = imagen_a_vector(self.imagen)
        if vector is None:
            messagebox.showwarning("Lienzo vacío", "Dibuje un dígito antes de reconocer.")
            return

        probs = self.modelo.predict(vector, verbose=0)[0]
        pred = int(np.argmax(probs))
        prob_max = float(probs[pred])
        top3 = np.argsort(probs)[::-1][:3]

        self._mostrar_8x8(img8)
        self.lbl_pred.config(text=str(pred))
        self.lbl_top3.config(text="Top 3:  " + "   ".join(
            f"{d}: {probs[d]:.1%}" for d in top3))
        self._dibujar_barras(probs, pred)

        baja = prob_max < UMBRAL_CONFIANZA
        self.lbl_aviso.config(
            text=(f"⚠ Baja confianza: la mayor probabilidad ({prob_max:.1%}) es menor "
                  f"que {UMBRAL_CONFIANZA:.0%}. La predicción puede ser incorrecta.")
            if baja else "")

        self.ultimo = {"prediccion": pred, "prob_maxima": prob_max,
                       "baja_confianza": baja, "img8": img8}
        self.btn_registrar.config(state="normal")
        self.lbl_estado.config(text=f"Reconocido: {pred} (p = {prob_max:.3f})")

    def _mostrar_8x8(self, img8):
        self.vista8.delete("all")
        celda = 20
        for f in range(8):
            for c in range(8):
                v = int(255 - img8[f, c] / 16 * 255)   # 0 = blanco, 16 = negro
                color = f"#{v:02x}{v:02x}{v:02x}"
                self.vista8.create_rectangle(c * celda, f * celda, (c + 1) * celda,
                                             (f + 1) * celda, fill=color, outline="#ddd")

    def _dibujar_barras(self, probs, pred=None):
        g = self.grafico
        g.delete("all")
        ancho, alto, base, margen = 380, 190, 165, 15
        paso = (ancho - 2 * margen) / 10
        alto_max = 130
        # Línea del umbral de confianza
        y_umbral = base - UMBRAL_CONFIANZA * alto_max
        g.create_line(margen, y_umbral, ancho - margen, y_umbral, fill="#c62828", dash=(4, 3))
        g.create_text(ancho - margen, y_umbral - 7, text="0.70", fill="#c62828",
                      anchor="e", font=("Arial", 7))
        for d in range(10):
            x0 = margen + d * paso + 5
            x1 = x0 + paso - 10
            h = probs[d] * alto_max
            color = "#1565c0" if d == pred else "#90caf9"
            g.create_rectangle(x0, base - h, x1, base, fill=color, outline="")
            g.create_text((x0 + x1) / 2, base + 12, text=str(d), font=("Arial", 9, "bold"))
            if probs[d] >= 0.01:
                g.create_text((x0 + x1) / 2, base - h - 8, text=f"{probs[d]:.0%}",
                              font=("Arial", 7))
        g.create_line(margen, base, ancho - margen, base)

    # ------------------------------------------------------------------
    # Historial
    # ------------------------------------------------------------------
    def registrar(self):
        if self.ultimo is None:
            messagebox.showwarning("Sin predicción", "Primero presione Reconocer.")
            return
        texto = self.entrada_etiqueta.get().strip()
        if not (len(texto) == 1 and texto.isdigit()):
            messagebox.showerror("Etiqueta inválida",
                                 "Escriba un solo dígito entre 0 y 9.\n"
                                 f"Valor ingresado: '{texto}'")
            return
        etiqueta = int(texto)
        es_prueba = self.var_prueba.get()

        # Prueba oficial: máximo 2 dibujos por dígito (20 en total)
        if es_prueba:
            ya = sum(1 for f in self._leer_historial()
                     if f["tipo"] == "prueba" and f["etiqueta"] == str(etiqueta))
            if ya >= DIBUJOS_POR_DIGITO:
                if not messagebox.askyesno(
                        "Dígito completo",
                        f"Ya hay {ya} dibujos de prueba del dígito {etiqueta}.\n"
                        "¿Registrarlo de todos modos como prueba?"):
                    return

        resultado = "correcto" if etiqueta == self.ultimo["prediccion"] else "incorrecto"

        nuevo = not os.path.exists(ARCHIVO_HISTORIAL)
        with open(ARCHIVO_HISTORIAL, "a", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            if nuevo:
                escritor.writerow(COLUMNAS)
            escritor.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                etiqueta,
                self.ultimo["prediccion"],
                f"{self.ultimo['prob_maxima']:.4f}",
                resultado,
                "si" if self.ultimo["baja_confianza"] else "no",
                "prueba" if es_prueba else "practica",
                " ".join(str(int(v)) for v in self.ultimo["img8"].ravel()),
            ])

        # Guardar el dibujo y su imagen 8x8 como evidencia verificable
        os.makedirs(CARPETA_DIBUJOS, exist_ok=True)
        numero = self._contar_registros()      # n.º de registro: nombre único
        base = os.path.join(CARPETA_DIBUJOS,
                            f"{numero:03d}_quiso{etiqueta}_pred{self.ultimo['prediccion']}")
        self.imagen.save(base + "_lienzo.png")
        img8 = (255 - self.ultimo["img8"] / 16 * 255).astype(np.uint8)
        Image.fromarray(img8).resize((160, 160), Image.NEAREST).save(base + "_8x8.png")

        # Un solo registro por predicción (se conserva el primer intento)
        self.btn_registrar.config(state="disabled")
        self._actualizar_historial()
        self.lbl_estado.config(
            text=f"Registrado: quiso {etiqueta}, predijo {self.ultimo['prediccion']} "
                 f"-> {resultado} | Registros en historial: {self._contar_registros()}")

    def _leer_historial(self):
        if not os.path.exists(ARCHIVO_HISTORIAL):
            return []
        with open(ARCHIVO_HISTORIAL, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _actualizar_historial(self):
        filas = self._leer_historial()
        # Tabla: últimos 5 registros (el más reciente arriba)
        self.tabla.delete(*self.tabla.get_children())
        for f in reversed(filas[-5:]):
            self.tabla.insert("", "end", tags=(f["resultado"],), values=(
                f["fecha_hora"], f["etiqueta"], f["prediccion"], f["prob_maxima"],
                f["resultado"], f["baja_confianza"], f["tipo"]))
        # Progreso de la prueba oficial
        prueba = [f for f in filas if f["tipo"] == "prueba"]
        for d, lbl in enumerate(self.lbl_digitos):
            n = sum(1 for f in prueba if f["etiqueta"] == str(d))
            color = "#a5d6a7" if n >= DIBUJOS_POR_DIGITO else ("#fff59d" if n else "SystemButtonFace")
            try:
                lbl.config(text=f"{d}\n{n}/{DIBUJOS_POR_DIGITO}", bg=color)
            except tk.TclError:
                lbl.config(text=f"{d}\n{n}/{DIBUJOS_POR_DIGITO}", bg="#eeeeee")
        correctos = sum(1 for f in prueba if f["resultado"] == "correcto")
        avisos = sum(1 for f in prueba if f["baja_confianza"] == "si")
        self.marco_prog.config(text=f"Prueba oficial: {len(prueba)}/{10 * DIBUJOS_POR_DIGITO}")
        texto = f"Correctos: {correctos} | Avisos de baja confianza: {avisos}"
        if prueba:
            texto += f" | Accuracy: {correctos / len(prueba):.0%}"
        self.lbl_resumen.config(text=texto)

    def _contar_registros(self):
        if not os.path.exists(ARCHIVO_HISTORIAL):
            return 0
        with open(ARCHIVO_HISTORIAL, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)


if __name__ == "__main__":
    raiz = tk.Tk()
    AppDigitos(raiz)
    raiz.mainloop()
