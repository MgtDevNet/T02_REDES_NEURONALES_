"""
Evalúa en prueba los modelos guardados de la Parte 1 y genera las figuras del reporte.

    python evaluar_pd.py                         # referencia del notebook, models/pd_logistica y models/pd
    python evaluar_pd.py --modelos models/pd     # referencia del notebook frente a un solo modelo

El modelo de referencia es la regresión logística de Punto_1/Primer_Punto.ipynb (sklearn, L2 con
C elegido por pérdida logarítmica en validación). Se usan las predicciones en prueba que guardó
el notebook en modelo_logistico.joblib, sin volver a predecir, así que no dependen de la versión
de scikit-learn. La logística de models/pd_logistica (nuestra red sin capas ocultas) solo aparece
en la tabla, como verificación de que el código en NumPy la reproduce; las figuras comparan la
referencia con las redes.

No reentrena: carga red.npz y preprocesador.joblib de cada carpeta. Genera:
    reports/evaluacion_pd.csv          métricas en prueba: total, dti <= 30 y dti > 30
    reports/comparacion_pd.csv         bootstrap pareado de la diferencia red − referencia (AUC y Brier)
    reports/figures/pd_roc.png         curvas ROC
    reports/figures/pd_calibracion.png curva de calibración (deciles de PD)
    reports/figures/pd_ks.png          distribución acumulada del puntaje en buenos y malos

Sobre dti > 30: el máximo de dti en entrenamiento es 29.99, pero Lending Club subió su tope
durante 2012 y el 6 % de la prueba tiene dti entre 30 y 35. En esa franja la red extrapola,
por eso las métricas se reportan con y sin esos créditos.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metricas
from src.red_neuronal import RedNeuronal
from src.scorecard import Scorecard

FIGURAS = Path("reports/figures")
# Paleta categórica validada (orden fijo: el color sigue al modelo, no a su posición)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
REFERENCIA = "Regresión logística (referencia)"


def cargar_modelo(carpeta):
    carpeta = Path(carpeta)
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    return {
        "nombre": "Regresión logística" if not metadata["config"]["red"]["capas_ocultas"]
                  else "MLP " + "-".join(map(str, metadata["config"]["red"]["capas_ocultas"])),
        "red": RedNeuronal.cargar(carpeta / "red.npz"),
        "preprocesador": joblib.load(carpeta / "preprocesador.joblib"),
        "metadata": metadata,
    }


def predecir(modelo, X):
    return modelo["red"].predecir(modelo["preprocesador"].transform(X))


def cargar_referencia(ruta_modelo, ruta_datos, y):
    """Predicciones en prueba de la logística del notebook, verificadas fila a fila contra y."""
    with warnings.catch_warnings():
        # el joblib trae el LogisticRegression; aquí solo se leen sus predicciones guardadas
        warnings.filterwarnings("ignore", message="Trying to unpickle")
        guardado = joblib.load(ruta_modelo)
    p = np.asarray(guardado["p_test"], dtype=float)
    y_notebook = joblib.load(ruta_datos)["y_test"].to_numpy()
    if len(p) != len(y) or not np.array_equal(y_notebook, y):
        raise ValueError("Las predicciones del notebook no están en el mismo orden que la prueba "
                         "de pd_limpio.parquet: vuelva a ejecutar Punto_1/Primer_Punto.ipynb")
    return {"p": p, "parametros": len(guardado["columnas"]) + 1}     # coeficientes + intercepto


def bootstrap_pareado(y, p_referencia, p_modelo, repeticiones=1000, semilla=2026):
    """Diferencia modelo − referencia en AUC y Brier, remuestreando los mismos créditos para
    los dos modelos. Los IC de cada modelo por separado se traslapan aunque la diferencia sea
    estable; el remuestreo pareado mide directamente la incertidumbre de la diferencia."""
    rng = np.random.default_rng(semilla)
    diferencias = {"auc": [], "brier": []}
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        for nombre, lista in diferencias.items():
            medida = getattr(metricas, nombre)
            lista.append(medida(y[i], p_modelo[i]) - medida(y[i], p_referencia[i]))
    filas = []
    for nombre, lista in diferencias.items():
        lista = np.asarray(lista)
        medida = getattr(metricas, nombre)
        # mayor AUC es mejor; menor Brier es mejor
        mejor = lista > 0 if nombre == "auc" else lista < 0
        filas.append({"metrica": nombre,
                      "diferencia": medida(y, p_modelo) - medida(y, p_referencia),
                      "ic95_inf": float(np.percentile(lista, 2.5)),
                      "ic95_sup": float(np.percentile(lista, 97.5)),
                      "prop_modelo_mejor": float(mejor.mean())})
    return filas


def curva_roc(y, p):
    orden = np.argsort(-p, kind="mergesort")
    y = np.asarray(y)[orden]
    tpr = np.r_[0, np.cumsum(y == 1) / np.sum(y == 1)]
    fpr = np.r_[0, np.cumsum(y == 0) / np.sum(y == 0)]
    return fpr, tpr


def estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=12)
    ax.set_xlabel(xlabel, color=TINTA_SUAVE)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE)
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_SUAVE)
    ax.tick_params(colors=TINTA_SUAVE)


def figura_roc(y, predicciones, ruta):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, linewidth=1, linestyle="--", label="Azar (AUC = 0.5)")
    for (nombre, p), color in zip(predicciones.items(), COLORES):
        fpr, tpr = curva_roc(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{nombre} (AUC = {metricas.auc(y, p):.3f})")
    estilo(ax, "Curva ROC en prueba (jul–dic 2012)",
           "Tasa de falsos positivos (buenos rechazados)", "Tasa de verdaderos positivos (malos detectados)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_calibracion(y, predicciones, ruta):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    limite = 0
    for (nombre, p), color in zip(predicciones.items(), COLORES):
        tabla = metricas.curva_calibracion(y, p, n_grupos=10)
        limite = max(limite, tabla["p_promedio"].max(), tabla["tasa_observada"].max())
        ax.plot(tabla["p_promedio"], tabla["tasa_observada"], color=color, linewidth=2,
                marker="o", markersize=6, label=f"{nombre} (Brier = {metricas.brier(y, p):.4f})")
    limite *= 1.08
    ax.plot([0, limite], [0, limite], color=TINTA_SUAVE, linewidth=1, linestyle="--",
            label="Calibración perfecta")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    estilo(ax, "Curva de calibración en prueba (deciles de PD)",
           "PD promedio predicha", "Tasa de malos observada")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_ks(y, puntaje, nombre, ruta):
    y = np.asarray(y)
    rejilla = np.linspace(np.percentile(puntaje, 0.5), np.percentile(puntaje, 99.5), 400)
    f_malos = np.searchsorted(np.sort(puntaje[y == 1]), rejilla, side="right") / np.sum(y == 1)
    f_buenos = np.searchsorted(np.sort(puntaje[y == 0]), rejilla, side="right") / np.sum(y == 0)
    i = int(np.argmax(f_malos - f_buenos))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(rejilla, f_malos, color=COLORES[1], linewidth=2, label="Malos (y = 1)")
    ax.plot(rejilla, f_buenos, color=COLORES[0], linewidth=2, label="Buenos (y = 0)")
    ax.vlines(rejilla[i], f_buenos[i], f_malos[i], color=TINTA, linewidth=1.5)
    ax.annotate(f"KS = {f_malos[i] - f_buenos[i]:.3f}\n(puntaje {rejilla[i]:.0f})",
                (rejilla[i], (f_malos[i] + f_buenos[i]) / 2), xytext=(8, 0),
                textcoords="offset points", va="center", color=TINTA)
    estilo(ax, f"Distribución acumulada del puntaje en prueba — {nombre}",
           "Puntaje de la scorecard", "Proporción acumulada")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="+", default=["models/pd_logistica", "models/pd"])
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    parser.add_argument("--referencia", default="Punto_1/modelo_logistico.joblib",
                        help="logística del notebook (predicciones en prueba)")
    parser.add_argument("--datos-referencia", default="Punto_1/datos_modelado.joblib",
                        help="matrices del notebook, para verificar que y_test está en el mismo orden")
    parser.add_argument("--repeticiones", type=int, default=1000, help="remuestreos del bootstrap pareado")
    args = parser.parse_args(argumentos)

    modelos = [cargar_modelo(c) for c in args.modelos]
    for m in modelos:
        if not m["metadata"]["config"]["red"]["capas_ocultas"]:
            m["nombre"] = "Regresión logística (NumPy)"     # se distingue de la referencia
    datos = pd.read_parquet(args.datos)
    test = datos[datos["particion"] == "test"]
    y = test["y"].to_numpy()
    referencia = cargar_referencia(args.referencia, args.datos_referencia, y)
    predicciones = {REFERENCIA: referencia["p"], **{m["nombre"]: predecir(m, test) for m in modelos}}
    parametros = {REFERENCIA: referencia["parametros"],
                  **{m["nombre"]: m["metadata"]["parametros_entrenables"] for m in modelos}}

    grupos = {"total": np.ones(len(test), bool),
              "dti <= 30": (test["dti"] <= 30).to_numpy(),
              "dti > 30": (test["dti"] > 30).to_numpy()}
    filas = []
    for nombre, p in predicciones.items():
        for grupo, mascara in grupos.items():
            filas.append({"modelo": nombre, "grupo": grupo, "parametros": parametros[nombre],
                          **metricas.resumen(y[mascara], p[mascara]),
                          "pd_promedio": float(p[mascara].mean())})
    tabla = pd.DataFrame(filas)
    Path("reports").mkdir(exist_ok=True)
    tabla.to_csv("reports/evaluacion_pd.csv", index=False)
    print(tabla[["modelo", "grupo", "n", "tasa_malos", "pd_promedio", "auc", "gini", "ks", "brier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # las redes frente a la referencia, con los mismos créditos remuestreados
    redes = [m["nombre"] for m in modelos if m["metadata"]["config"]["red"]["capas_ocultas"]]
    comparacion = pd.DataFrame([
        {"modelo": nombre, "referencia": REFERENCIA, "repeticiones": args.repeticiones, **fila}
        for nombre in redes
        for fila in bootstrap_pareado(y, referencia["p"], predicciones[nombre], args.repeticiones)])
    if not comparacion.empty:
        comparacion.to_csv("reports/comparacion_pd.csv", index=False)
        print(f"\nDiferencia red − referencia en prueba (bootstrap pareado, {args.repeticiones} remuestreos):")
        print(comparacion[["modelo", "metrica", "diferencia", "ic95_inf", "ic95_sup", "prop_modelo_mejor"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # las figuras solo comparan la referencia con las redes
    en_figuras = {nombre: predicciones[nombre] for nombre in [REFERENCIA, *redes]}
    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_roc(y, en_figuras, FIGURAS / "pd_roc.png")
    figura_calibracion(y, en_figuras, FIGURAS / "pd_calibracion.png")
    ultimo = modelos[-1]
    scorecard = Scorecard(**ultimo["metadata"]["config"]["scorecard"])
    figura_ks(y, scorecard.puntaje(predicciones[ultimo["nombre"]]), ultimo["nombre"],
              FIGURAS / "pd_ks.png")
    print(f"\nFiguras en {FIGURAS}/  y métricas en reports/evaluacion_pd.csv y reports/comparacion_pd.csv")


if __name__ == "__main__":
    main()
