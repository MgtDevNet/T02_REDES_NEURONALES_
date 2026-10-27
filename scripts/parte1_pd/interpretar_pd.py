"""
Qué variables hacen más riesgosa a una persona (tarea 7 de la Parte 1).

    python -m scripts.parte1_pd.interpretar_pd                       # modelo de models/pd
    python -m scripts.parte1_pd.interpretar_pd --modelo models/pd_logistica --repeticiones 5

Dos métodos que funcionan con cualquier modelo (también con la red), sobre el conjunto de prueba:

1. Importancia por permutación (Breiman, 2001; Fisher, Rudin y Dominici, 2019).
   Se desordena al azar una variable, rompiendo su relación con y, y se mide cuánto cae el AUC.
   Si el modelo no usa la variable, el AUC no cambia. Se repite varias veces y se reporta la
   media y la desviación. Las variables que se derivan de la misma columna original se
   permutan juntas (p. ej. mths_since_last_delinq con nunca_mora), porque por separado
   quedarían combinaciones imposibles.

2. Dependencia parcial (Friedman, 2001).
   Para cada valor v de una variable, se le pone v a TODOS los créditos de prueba y se promedia
   la PD predicha. La curva dice en qué dirección mueve el riesgo la variable, en promedio,
   manteniendo las demás como están en los datos.

Genera reports/importancia_pd.csv, reports/figures/pd_importancia.png y
reports/figures/pd_dependencia.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.parte1_pd.evaluar_pd import COLORES, FIGURAS, TINTA_SUAVE, cargar_modelo, estilo, predecir
from src import metricas
from src.preprocesamiento import INDICADOR_DE
from src.variables import NOMBRES

# Columnas que se permutan juntas porque salen de la misma variable original
GRUPOS = {variable: [variable, indicador] for indicador, variable in INDICADOR_DE.items()}


def importancia_permutacion(modelo, X, y, repeticiones, semilla):
    rng = np.random.default_rng(semilla)
    base = metricas.auc(y, predecir(modelo, X))
    v = modelo["metadata"]["variables_entrada"]
    agrupadas = {c for cols in GRUPOS.values() for c in cols}
    variables = [c for c in v["numericas"] + v["categoricas"] if c not in agrupadas] + list(GRUPOS)

    filas = []
    for var in variables:
        columnas = GRUPOS.get(var, [var])
        caidas = []
        for _ in range(repeticiones):
            Xp = X.copy()
            orden = rng.permutation(len(X))
            Xp[columnas] = X[columnas].to_numpy()[orden]
            caidas.append(base - metricas.auc(y, predecir(modelo, Xp)))
        filas.append({"variable": var, "nombre": NOMBRES.get(var, var),
                      "caida_auc": np.mean(caidas), "desviacion": np.std(caidas, ddof=1)})
    return base, pd.DataFrame(filas).sort_values("caida_auc", ascending=False)


def dependencia_parcial(modelo, X, variable, n_puntos=20):
    valores = np.unique(np.quantile(X[variable].dropna(), np.linspace(0.02, 0.98, n_puntos)))
    pd_media = []
    for v in valores:
        Xv = X.copy()
        Xv[variable] = v
        if variable in GRUPOS:            # un valor observado implica que el indicador es 0
            Xv[GRUPOS[variable][1]] = 0
        pd_media.append(predecir(modelo, Xv).mean())
    return valores, np.array(pd_media)


def figura_importancia(tabla, base, nombre, ruta):
    t = tabla.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(t["nombre"], t["caida_auc"], xerr=t["desviacion"], color=COLORES[0], height=0.6,
            error_kw={"ecolor": TINTA_SUAVE, "elinewidth": 1})
    estilo(ax, f"Importancia por permutación en prueba — {nombre}",
           f"Caída del AUC al desordenar la variable (AUC sin permutar = {base:.3f})", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_dependencia(modelo, X, variables, ruta):
    columnas = 3
    filas = int(np.ceil(len(variables) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(11, 3.4 * filas), sharey=True)
    for ax, var in zip(np.ravel(ejes), variables):
        valores, pd_media = dependencia_parcial(modelo, X, var)
        ax.plot(valores, 100 * pd_media, color=COLORES[1], linewidth=2)
        estilo(ax, NOMBRES.get(var, var), "", "PD promedio (%)" if ax in np.ravel(ejes)[::columnas] else "")
        if valores.max() >= 10_000:       # montos en miles para que no se encimen
            ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    for ax in np.ravel(ejes)[len(variables):]:
        ax.axis("off")
    fig.suptitle(f"Dependencia parcial en prueba — {modelo['nombre']}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="models/pd")
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args(argumentos)

    modelo = cargar_modelo(args.modelo)
    datos = pd.read_parquet(args.datos)
    test = datos[datos["particion"] == "test"].reset_index(drop=True)
    y = test["y"].to_numpy()

    base, tabla = importancia_permutacion(modelo, test, y, args.repeticiones, args.semilla)
    Path("reports").mkdir(exist_ok=True)
    tabla.to_csv("reports/importancia_pd.csv", index=False)
    print(f"AUC en prueba sin permutar: {base:.4f}\n")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_importancia(tabla, base, modelo["nombre"], FIGURAS / "pd_importancia.png")
    numericas = [v for v in tabla["variable"] if v not in modelo["metadata"]["variables_entrada"]["categoricas"]]
    # dti siempre se muestra: en prueba hay valores (30-35) que no existen en entrenamiento
    graficar = numericas[:5] + ["dti"] if "dti" not in numericas[:5] else numericas[:6]
    figura_dependencia(modelo, test, graficar, FIGURAS / "pd_dependencia.png")
    print(f"\nFiguras en {FIGURAS}/")


if __name__ == "__main__":
    main()
