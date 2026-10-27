"""
Búsqueda aleatoria de hiperparámetros para la Parte 1 (PD).

    python -m scripts.parte1_pd.buscar_hiperparametros_pd                     # 40 configuraciones de MLP + logística
    python -m scripts.parte1_pd.buscar_hiperparametros_pd --n 10 --demo       # versión rápida

Por qué búsqueda aleatoria: con varios hiperparámetros, solo unos pocos importan de verdad, y
el muestreo aleatorio prueba más valores distintos de cada uno que una grilla con el mismo
número de entrenamientos (Bergstra y Bengio, 2012).

Cada configuración se elige SOLO con el conjunto de validación (AUC; el desempate es la
entropía cruzada). Este script no guarda métricas de prueba: la prueba se usa una sola vez,
con la configuración final, en scripts/parte1_pd/repetir_semillas_pd.py.

Para la comparación justa del enunciado, la regresión logística también se ajusta: se prueba
su penalización L2 en una grilla corta. Ambas familias usan los mismos datos, preprocesamiento
y particiones. Se reporta cuántas configuraciones se entrenaron y el tiempo total.

Resultado: reports/busqueda_pd.csv (una fila por configuración) y la mejor de cada familia
impresa como comando --set listo para copiar.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.parte1_pd.entrenar_pd import entrenar
from src.configuracion import cargar_config

# Espacio de búsqueda del MLP
ARQUITECTURAS = [[8], [16], [32], [64], [16, 8], [32, 16], [64, 32], [64, 32, 16]]
ACTIVACIONES = ["relu", "tanh"]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]
TAMANOS_LOTE = [256, 512, 1024]
L2_LOGISTICA = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


def muestrear_mlp(rng):
    """Una configuración al azar. L2 y la tasa se muestrean en escala logarítmica porque
    lo que importa es su orden de magnitud (0.0001 vs 0.001), no su valor exacto."""
    optimizador = str(rng.choice(["adam", "sgd"], p=[0.75, 0.25]))
    rango_tasa = (-4, -2) if optimizador == "adam" else (-3, -1)
    return {
        "red.capas_ocultas": ARQUITECTURAS[rng.integers(len(ARQUITECTURAS))],
        "red.activacion": str(rng.choice(ACTIVACIONES)),
        "red.l2": float(10 ** rng.uniform(-6, -2)),
        "red.dropout": float(rng.choice(DROPOUTS)),
        "entrenamiento.optimizador": optimizador,
        "entrenamiento.tasa": float(10 ** rng.uniform(*rango_tasa)),
        "entrenamiento.tam_lote": int(rng.choice(TAMANOS_LOTE)),
    }


def configuraciones_logistica():
    return [{"red.capas_ocultas": [], "red.dropout": 0.0, "red.l2": l2} for l2 in L2_LOGISTICA]


def aplicar(config, cambios):
    config = copy.deepcopy(config)
    for clave, valor in cambios.items():
        seccion, nombre = clave.split(".")
        config[seccion][nombre] = valor
    return config


def como_set(cambios):
    partes = []
    for clave, valor in cambios.items():
        if isinstance(valor, list):
            valor = ",".join(map(str, valor))
        elif isinstance(valor, float):
            valor = f"{valor:.3g}"
        partes.append(f"--set {clave}={valor}")
    return " ".join(partes)


def evaluar(config, cambios, familia, demo):
    red, _, _, meta = entrenar(aplicar(config, cambios), demo=demo, verbose=False)
    v = meta["metricas"]["valid"]
    fila = {"familia": familia, **{k: str(x) if isinstance(x, list) else x for k, x in cambios.items()},
            "parametros": meta["parametros_entrenables"], "mejor_epoca": meta["mejor_epoca"],
            "segundos": meta["segundos_entrenamiento"],
            "auc_valid": v["auc"], "ks_valid": v["ks"], "brier_valid": v["brier"],
            "entropia_valid": v["entropia_cruzada"], "auc_train": meta["metricas"]["train"]["auc"]}
    return fila, cambios


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--n", type=int, default=40, help="configuraciones de MLP a probar")
    parser.add_argument("--demo", action="store_true", help="usa una muestra de los datos")
    parser.add_argument("--salida", default="reports/busqueda_pd.csv")
    args = parser.parse_args(argumentos)

    config = cargar_config(args.config, "pd")
    rng = np.random.default_rng(config["semilla"])
    candidatos = ([("logistica", c) for c in configuraciones_logistica()]
                  + [("mlp", muestrear_mlp(rng)) for _ in range(args.n)])

    inicio = time.perf_counter()
    filas, cambios_por_fila = [], []
    for i, (familia, cambios) in enumerate(candidatos, 1):
        fila, cambios = evaluar(config, cambios, familia, args.demo)
        filas.append(fila)
        cambios_por_fila.append(cambios)
        print(f"[{i:3d}/{len(candidatos)}] {familia:9s} AUC valid {fila['auc_valid']:.4f}"
              f"  ({fila['segundos']:.1f} s)  {como_set(cambios)}")
    total = time.perf_counter() - inicio

    tabla = pd.DataFrame(filas)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False)

    print(f"\nTiempo total: {total / 60:.1f} min   ->  {salida}")
    print(tabla.groupby("familia").agg(configuraciones=("auc_valid", "size"),
                                       minutos=("segundos", lambda s: s.sum() / 60),
                                       mejor_auc_valid=("auc_valid", "max")).to_string())
    for familia in ("logistica", "mlp"):
        sub = tabla[tabla["familia"] == familia]
        mejor = sub.sort_values(["auc_valid", "entropia_valid"], ascending=[False, True]).index[0]
        print(f"\nMejor {familia}: AUC valid {tabla.loc[mejor, 'auc_valid']:.4f}, "
              f"{tabla.loc[mejor, 'parametros']} parámetros")
        print(f"  {como_set(cambios_por_fila[mejor])}")


if __name__ == "__main__":
    main()
