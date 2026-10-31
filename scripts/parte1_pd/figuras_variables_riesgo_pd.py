"""
Variables de riesgo con la regresión logística, comparada con la red (Parte 1).

    python -m scripts.parte1_pd.figuras_variables_riesgo_pd

Aplica los métodos de interpretar_pd.py (importancia por permutación y dependencia parcial, en
prueba) a la logística de models/pd_logistica y a la red de models/pd, con la misma semilla y
el mismo número de repeticiones. No sobrescribe lo que genera interpretar_pd.py. Genera:
    reports/importancia_pd_logistica.csv              importancia por permutación de la logística
    reports/figures/pd_importancia_comparada.png      logística (barras) frente a la red (puntos)
    reports/figures/pd_dependencia_logistica.png      dependencia parcial de la logística
y muestra la PD promedio en los extremos de cada curva para los dos modelos.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.parte1_pd.evaluar_pd import FIGURAS, cargar_modelo
from scripts.parte1_pd.interpretar_pd import dependencia_parcial, figura_dependencia, importancia_permutacion

AZUL, NARANJA, TINTA_SUAVE = "#2f7ed8", "#f0663a", "#666666"


def figura_comparada(logistica, red, ruta):
    logistica = logistica.set_index("nombre").sort_values("caida_auc")
    red = red.set_index("nombre").loc[logistica.index]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    filas = np.arange(len(logistica))
    ax.barh(filas, logistica["caida_auc"], xerr=logistica["desviacion"], color=AZUL, height=0.6,
            label="Regresión logística", error_kw={"ecolor": TINTA_SUAVE, "elinewidth": 1})
    ax.plot(red["caida_auc"], filas, "o", color=NARANJA, ms=6, label="Red neuronal MLP 64-32", zorder=3)
    ax.set_yticks(filas)
    ax.set_yticklabels(logistica.index)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("Caída del AUC en prueba al desordenar la variable")
    ax.set_title("Importancia por permutación", loc="left", fontsize=10.5)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e6e6e6")
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logistica", default="models/pd_logistica")
    parser.add_argument("--red", default="models/pd")
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    parser.add_argument("--repeticiones", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args(argumentos)

    datos = pd.read_parquet(args.datos)
    test = datos[datos["particion"] == "test"].reset_index(drop=True)
    y = test["y"].to_numpy()
    modelos = {"logistica": cargar_modelo(args.logistica), "red": cargar_modelo(args.red)}

    tablas = {}
    for clave, modelo in modelos.items():
        base, tablas[clave] = importancia_permutacion(modelo, test, y, args.repeticiones, args.semilla)
        print(f"{modelo['nombre']}: AUC en prueba sin permutar {base:.4f}")
    tablas["logistica"].to_csv("reports/importancia_pd_logistica.csv", index=False)
    comparacion = tablas["logistica"].merge(tablas["red"], on=["variable", "nombre"], suffixes=("_logistica", "_red"))
    print(comparacion[["nombre", "caida_auc_logistica", "caida_auc_red"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_comparada(tablas["logistica"], tablas["red"], FIGURAS / "pd_importancia_comparada.png")

    logistica = modelos["logistica"]
    categoricas = logistica["metadata"]["variables_entrada"]["categoricas"]
    numericas = [v for v in tablas["logistica"]["variable"] if v not in categoricas]
    graficar = numericas[:5] + ["dti"] if "dti" not in numericas[:5] else numericas[:6]
    figura_dependencia(logistica, test, graficar, FIGURAS / "pd_dependencia_logistica.png")

    print("\nPD promedio (%) en los extremos de la dependencia parcial (percentiles 2 y 98):")
    for variable in graficar:
        extremos = []
        for modelo in modelos.values():
            valores, pd_media = dependencia_parcial(modelo, test, variable)
            extremos.append(f"{100 * pd_media[0]:.1f} → {100 * pd_media[-1]:.1f}")
        print(f"  {variable:18s} valores {valores[0]:g} → {valores[-1]:g}   logística {extremos[0]}   red {extremos[1]}")
    print(f"\nFiguras en {FIGURAS}/")


if __name__ == "__main__":
    main()
