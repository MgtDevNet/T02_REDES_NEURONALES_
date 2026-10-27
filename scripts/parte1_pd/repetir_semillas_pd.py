"""
Entrena una configuración con varias semillas y resume la variabilidad de sus métricas.

    python -m scripts.parte1_pd.repetir_semillas_pd                                   # config.ini (MLP), 30 semillas
    python -m scripts.parte1_pd.repetir_semillas_pd --nombre logistica \
        --set red.capas_ocultas= --set red.dropout=0 --set red.l2=0  # la regresión logística
    python -m scripts.parte1_pd.repetir_semillas_pd --semillas 5 --demo               # versión rápida

El enunciado pide al menos 30 corridas independientes de la configuración final de cada
modelo, reportando media, desviación estándar, mejor y peor valor de cada métrica. La semilla
cambia la inicialización de los pesos, el orden de los mini-lotes y las máscaras de dropout;
los datos, el preprocesamiento y las particiones son siempre los mismos.

Resultado: reports/semillas_pd_<nombre>.csv (una fila por semilla) y el resumen en pantalla.
"""

import argparse
import copy
import time
from pathlib import Path

import pandas as pd

from scripts.parte1_pd.entrenar_pd import PARTICIONES, entrenar
from src.configuracion import cargar_config

METRICAS = ["auc", "gini", "ks", "brier"]


def resumir(tabla, particion):
    """Media, desviación, mejor y peor valor. Para el Brier, menor es mejor."""
    filas = {}
    for m in METRICAS:
        s = tabla[f"{m}_{particion}"]
        mejor, peor = (s.min(), s.max()) if m == "brier" else (s.max(), s.min())
        filas[m] = {"media": s.mean(), "desviacion": s.std(ddof=1), "mejor": mejor, "peor": peor}
    return pd.DataFrame(filas).T


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--semillas", type=int, default=30)
    parser.add_argument("--nombre", default="mlp", help="nombre del modelo en el archivo de salida")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argumentos)

    config = cargar_config(args.config, "pd", args.set)
    inicio = time.perf_counter()
    filas = []
    for i in range(args.semillas):
        semilla = config["semilla"] + i
        c = copy.deepcopy(config)
        c["semilla"] = semilla
        _, _, _, meta = entrenar(c, demo=args.demo, verbose=False)
        fila = {"semilla": semilla, "mejor_epoca": meta["mejor_epoca"],
                "segundos": meta["segundos_entrenamiento"]}
        for p in PARTICIONES:
            fila.update({f"{m}_{p}": meta["metricas"][p][m] for m in METRICAS})
        filas.append(fila)
        print(f"semilla {semilla:3d}  AUC valid {fila['auc_valid']:.4f}  test {fila['auc_test']:.4f}")
    total = time.perf_counter() - inicio

    tabla = pd.DataFrame(filas)
    salida = Path(f"reports/semillas_pd_{args.nombre}.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False)

    print(f"\n{args.nombre}: {meta['parametros_entrenables']} parámetros entrenables, "
          f"{args.semillas} semillas, {total / 60:.1f} min  ->  {salida}")
    for p in ("valid", "test"):
        print(f"\n[{p}]")
        print(resumir(tabla, p).to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
