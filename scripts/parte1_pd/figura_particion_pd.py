"""
Línea de tiempo de la partición temporal de la Parte 1 (solo créditos a 36 meses).

    python -m scripts.parte1_pd.figura_particion_pd

Las fechas y los conteos son los de la partición del notebook (Punto_1/Primer_Punto.ipynb):
entrenamiento hasta dic 2011, validación ene–jun 2012, prueba jul–dic 2012; 2013–2015 se
excluye porque esos créditos seguían vigentes al cierre de los datos. Genera:
    reports/figures/pd_particion.png
"""
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

salida = "reports/figures/pd_particion.png"

segmentos = [  # inicio, fin, color, nombre, líneas de detalle, posición del texto
    (date(2007, 6, 1), date(2012, 1, 1), "#2f7ed8", "Entrenamiento",
     ["jun 2007 – dic 2011", "31.505 créditos · 12,3 % malos"], "abajo"),
    (date(2012, 1, 1), date(2012, 7, 1), "#f0663a", "Validación · ene–jun 2012",
     ["14.967 créditos · 13,8 % malos"], "arriba"),
    (date(2012, 7, 1), date(2013, 1, 1), "#2aa876", "Prueba",
     ["jul–dic 2012", "28.458 créditos · 13,6 % malos"], "abajo"),
    (date(2013, 1, 1), date(2016, 1, 1), "#bdbdbd", "Excluidos: 2013–2015",
     ["préstamos aún sin terminar", "(desenlace censurado)"], "abajo"),
]

fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
y0, alto = 0.0, 1.0
for inicio, fin, color, nombre, lineas, lugar in segmentos:
    x0, x1 = mdates.date2num(inicio), mdates.date2num(fin)
    excluido = nombre.startswith("Excluidos")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, alto, facecolor=color,
                           alpha=0.25 if excluido else 0.35, edgecolor=color if not excluido else "#888888",
                           linestyle="--" if excluido else "-", linewidth=1.4))
    xc = (x0 + x1) / 2 + (mdates.date2num(date(2013, 6, 1)) - mdates.date2num(date(2013, 1, 1)) if excluido else 0)
    if lugar == "abajo":
        ax.text(xc, y0 - 0.35, nombre, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(xc, y0 - 0.75, "\n".join(lineas), ha="center", va="top", fontsize=9.5,
                color="#555555", linespacing=1.5)
    else:
        ax.text(xc, y0 + alto + 0.95, nombre, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(xc, y0 + alto + 0.55, "\n".join(lineas), ha="center", va="bottom", fontsize=9.5,
                color="#555555")

for anio in range(2008, 2016):
    x = mdates.date2num(date(anio, 1, 1))
    ax.plot([x, x], [y0 - 0.08, y0 + alto + 0.08], color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.text(x, y0 + alto + 0.15, str(anio), ha="center", va="bottom", fontsize=9, color="#555555")

ax.set_xlim(mdates.date2num(date(2007, 4, 1)), mdates.date2num(date(2016, 3, 1)))
ax.set_ylim(-1.75, 2.6)
ax.axis("off")
ax.set_title("Partición temporal por fecha de originación (solo créditos a 36 meses)",
             loc="left", fontsize=12.5, pad=4)
fig.tight_layout()
fig.savefig(salida, bbox_inches="tight", facecolor="white")
print(salida)
