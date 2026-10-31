"""
Figuras de la scorecard (Parte 1) para el informe.

    python -m scripts.parte1_pd.figuras_scorecard_pd                         # regresión logística (models/pd_logistica)
    python -m scripts.parte1_pd.figuras_scorecard_pd --modelo models/pd      # la red neuronal

No reentrena: carga el modelo guardado y los parámetros de su scorecard. Genera:
    reports/figures/scorecard_transformacion.png  puntaje frente a ln(odds) y frente a la PD
    reports/figures/scorecard_rango.png           puntos que mueve cada variable (percentiles 5 a 95)
    reports/figures/scorecard_ejemplo.png         cascada de puntos de un cliente de prueba (PD mediana)
    reports/figures/scorecard_distribucion.png    puntajes de buenos y malos en prueba
y muestra la tasa de malos por banda de puntaje en prueba.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.parte1_pd.evaluar_pd import cargar_modelo
from scripts.parte1_pd.scorecard_pd import tabla_scorecard
from src.scorecard import Scorecard, puntos_por_variable
from src.variables import NOMBRES

AZUL, NARANJA, VERDE, GRIS = "#2f7ed8", "#f0663a", "#2aa876", "#9e9e9e"
TINTA, TINTA_SUAVE = "#222222", "#666666"
CARPETA = "reports/figures"


def estilo():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#999999", "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE})


def coma(x, decimales=1):
    return f"{x:.{decimales}f}".replace(".", ",")


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"{CARPETA}/{nombre}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura_transformacion(scorecard, puntajes):
    s = scorecard.parametros()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ln_odds = np.linspace(np.log(0.5), np.log(400), 200)
    ax[0].plot(ln_odds, s["offset"] + s["factor"] * ln_odds, color=AZUL, lw=2)
    for odds in (s["odds_base"] / 2, s["odds_base"], s["odds_base"] * 2):
        v = s["offset"] + s["factor"] * np.log(odds)
        ax[0].plot(np.log(odds), v, "o", color=NARANJA, ms=7, zorder=3)
        ax[0].annotate(f"odds {odds:g}:1 → {v:.0f}", (np.log(odds), v), xytext=(-95, 8),
                       textcoords="offset points", fontsize=9, color=TINTA)
    ax[0].set_xlabel("ln(odds buenos:malos)")
    ax[0].set_ylabel("Puntaje")
    ax[0].set_title(f"Recta: puntaje = {coma(s['offset'])} + {coma(s['factor'], 2)} · ln(odds)",
                    loc="left", fontsize=10.5)

    pd_ = np.linspace(0.005, 0.45, 300)
    ax[1].plot(pd_ * 100, scorecard.puntaje(pd_), color=AZUL, lw=2)
    p1, p99 = np.percentile(puntajes, [1, 99])
    ax[1].axhspan(p1, p99, color="#bdbdbd", alpha=0.3, lw=0)
    ax[1].text(30, p99 + 3, "98 % de la prueba", fontsize=9, color=TINTA_SUAVE)
    for q in (0.02, 0.05, 0.136, 0.30):
        v = float(scorecard.puntaje(q))
        ax[1].plot(q * 100, v, "o", color=NARANJA, ms=7, zorder=3)
        ax[1].annotate(f"PD {coma(q * 100)} % → {v:.0f}", (q * 100, v), xytext=(8, 4),
                       textcoords="offset points", fontsize=9)
    ax[1].set_xlabel("PD (%)")
    ax[1].set_ylabel("Puntaje")
    ax[1].set_title("Misma regla vista desde la PD", loc="left", fontsize=10.5)
    guardar(fig, "scorecard_transformacion.png")


def figura_rango(tabla):
    rango = tabla.groupby("nombre")["puntos"].agg(["min", "max"])
    rango = rango.assign(ancho=rango["max"] - rango["min"]).sort_values("ancho")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    filas = np.arange(len(rango))
    ax.hlines(filas, rango["min"], rango["max"], color=AZUL, lw=4, alpha=0.8)
    ax.plot(rango["min"], filas, "o", color=NARANJA, ms=6, label="valor que más resta")
    ax.plot(rango["max"], filas, "o", color=VERDE, ms=6, label="valor que más suma")
    ax.axvline(0, color="#999999", lw=1)
    ax.set_yticks(filas)
    ax.set_yticklabels(rango.index)
    ax.set_xlabel("Puntos frente al solicitante promedio (numéricas: percentiles 5 a 95 de entrenamiento)")
    ax.set_title("Cuántos puntos puede mover cada variable", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    guardar(fig, "scorecard_rango.png")


def figura_ejemplo(base, puntos):
    """Cascada desde el solicitante promedio hasta el puntaje del cliente."""
    principales = puntos[puntos.abs() >= 0.5].sort_values(key=abs, ascending=False)
    pasos = list(principales.values) + [puntos.sum() - principales.sum()]
    etiquetas = (["Solicitante\npromedio"] + [NOMBRES.get(v, v) for v in principales.index]
                 + ["Resto", "Puntaje\nfinal"])
    final = base + puntos.sum()
    piso = np.floor(min(base, final, base + np.cumsum(pasos).min()) / 5) * 5 - 10

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(0, base - piso, bottom=piso, color=GRIS, width=0.6)
    ax.text(0, base + 1, coma(base), ha="center", fontsize=8.5)
    acumulado = base
    for k, v in enumerate(pasos, start=1):
        ax.bar(k, v, bottom=acumulado, color=VERDE if v > 0 else NARANJA, width=0.6)
        ax.text(k, max(acumulado, acumulado + v) + 0.6, f"{v:+.1f}".replace(".", ","), ha="center", fontsize=8.5)
        acumulado += v
    n = len(etiquetas) - 1
    ax.bar(n, final - piso, bottom=piso, color=AZUL, width=0.6)
    ax.text(n, final + 1, coma(final), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=8, rotation=35, ha="right")
    ax.set_ylim(piso, max(base, final, base + np.cumsum(pasos).max()) + 12)
    ax.set_ylabel("Puntaje")
    ax.set_title("Un cliente de prueba: del solicitante promedio a su puntaje", loc="left", fontsize=10.5)
    guardar(fig, "scorecard_ejemplo.png")


def figura_distribucion(puntajes, y):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = np.arange(np.floor(puntajes.min() / 4) * 4, puntajes.max() + 4, 4)
    for clase, color, nombre in ((0, AZUL, "Buenos (pagaron)"), (1, NARANJA, "Malos (no pagaron)")):
        ax.hist(puntajes[y == clase], bins=bins, density=True, color=color, alpha=0.45,
                label=nombre, edgecolor="white", linewidth=1)
    media_malos, media_buenos = puntajes[y == 1].mean(), puntajes[y == 0].mean()
    ax.axvline(media_malos, color=NARANJA, lw=1.5, ls="--")
    ax.axvline(media_buenos, color=AZUL, lw=1.5, ls="--")
    alto = ax.get_ylim()[1] * 0.95
    ax.text(media_malos - 1, alto, f"media malos {media_malos:.0f}", ha="right", fontsize=9, color=TINTA)
    ax.text(media_buenos + 1, alto, f"media buenos {media_buenos:.0f}", ha="left", fontsize=9, color=TINTA)
    ax.set_xlabel("Puntaje (más puntos = menos riesgo)")
    ax.set_ylabel("Densidad")
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.88))
    ax.set_title("Puntajes en prueba para buenos y malos", loc="left", fontsize=10.5)
    guardar(fig, "scorecard_distribucion.png")


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="models/pd_logistica")
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    args = parser.parse_args(argumentos)

    modelo = cargar_modelo(args.modelo)
    scorecard = Scorecard(**modelo["metadata"]["config"]["scorecard"])
    pre = modelo["preprocesador"]
    referencia = np.array(modelo["metadata"]["referencia_transformada"])
    datos = pd.read_parquet(args.datos)
    train = datos[datos["particion"] == "train"]
    test = datos[datos["particion"] == "test"].reset_index(drop=True)

    X = pre.transform(test)
    p = modelo["red"].predecir(X).ravel()
    puntajes = scorecard.puntaje(p)
    y = test["y"].to_numpy()

    estilo()
    figura_transformacion(scorecard, puntajes)
    figura_rango(tabla_scorecard(modelo, scorecard, train))
    i = int(np.argsort(puntajes)[len(puntajes) // 2])          # cliente con el puntaje mediano
    base, puntos, _ = puntos_por_variable(modelo["red"], scorecard, X[[i]], referencia, pre.grupos())
    figura_ejemplo(base, puntos.iloc[0])
    figura_distribucion(puntajes, y)

    bandas = pd.cut(puntajes, [-np.inf, 520, 530, 540, 550, 560, 570, np.inf], right=False)
    resumen = pd.DataFrame({"banda": bandas, "y": y}).groupby("banda", observed=True)["y"].agg(["size", "mean"])
    print(f"Modelo: {modelo['nombre']}   cliente de ejemplo: fila {i} de prueba, PD {p[i]:.4f}, puntaje {puntajes[i]:.1f}")
    print("Tasa de malos por banda de puntaje en prueba:")
    print(resumen.rename(columns={"size": "creditos", "mean": "tasa_malos"}).round(3).to_string())
    print(f"Figuras en {CARPETA}/scorecard_*.png")


if __name__ == "__main__":
    main()
