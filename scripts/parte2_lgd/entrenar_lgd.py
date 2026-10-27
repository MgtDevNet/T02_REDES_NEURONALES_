"""
Entrena el modelo de pérdida dado incumplimiento (LGD) con un solo comando.

    python -m scripts.parte2_lgd.entrenar_lgd                                    # usa config.ini
    python -m scripts.parte2_lgd.entrenar_lgd --demo                             # muestra de los datos (< 2 min)
    python -m scripts.parte2_lgd.entrenar_lgd --set red.perdida=entropia_cruzada  # respuesta fraccional
    python -m scripts.parte2_lgd.entrenar_lgd --set red.capas_ocultas= --set red.salida=identidad \
                           --set red.dropout=0 --set salida.carpeta=models/lgd_lineal

Los hiperparámetros se cambian en config.ini (secciones [lgd.xxx]) o con --set.

Guarda en la carpeta de salida:
    red.npz                  pesos y arquitectura de la red
    preprocesador.joblib     imputación, escalado y one-hot ajustados con train
    predicciones_test.npz    LGD observada y predicha en test, para las figuras
    metadata.json            variables, configuración, métricas, distribuciones y tiempos
"""

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.parte1_pd.entrenar_pd import PARTICIONES, cargar_datos
from src import metricas
from src.configuracion import cargar_config, imprimir_config
from src.preprocesamiento import construir_preprocesador
from src.red_neuronal import RedNeuronal


def entrenar(config, demo=False, verbose=True):
    inicio = time.perf_counter()
    d, r, e = config["datos"], config["red"], config["entrenamiento"]

    datos = cargar_datos(config, demo)
    faltantes = set(PARTICIONES) - set(datos)
    if faltantes:
        raise ValueError(f"Faltan particiones en los datos: {sorted(faltantes)}")

    y = {nombre: datos[nombre][1] for nombre in PARTICIONES}
    recortados = {nombre: 0 for nombre in PARTICIONES}
    if d.get("recortar_objetivo"):
        minimo, maximo = d["recortar_objetivo"]
        for nombre in PARTICIONES:
            recortados[nombre] = int(np.sum((y[nombre] < minimo) | (y[nombre] > maximo)))
            y[nombre] = np.clip(y[nombre], minimo, maximo)

    preprocesador = construir_preprocesador(d["numericas"], d["categoricas"], d["logaritmicas"])
    X = {"train": preprocesador.fit_transform(datos["train"][0])}
    X.update({nombre: preprocesador.transform(datos[nombre][0]) for nombre in ("valid", "test")})

    red = RedNeuronal(X["train"].shape[1], capas_ocultas=r["capas_ocultas"],
                      activacion=r["activacion"], salida=r["salida"], perdida=r["perdida"],
                      l2=r["l2"], dropout=r["dropout"], semilla=config["semilla"])
    red.entrenar(X["train"], y["train"], X["valid"], y["valid"],
                 optimizador=e["optimizador"], tasa=e["tasa"], epocas=e["epocas"],
                 tam_lote=e["tam_lote"], paciencia=e["paciencia"], verbose=verbose)
    segundos = time.perf_counter() - inicio

    y_hat = {nombre: red.predecir(X[nombre]) for nombre in PARTICIONES}
    resultados = {nombre: metricas.resumen_regresion(y[nombre], y_hat[nombre])
                  for nombre in PARTICIONES}
    distribucion_test = metricas.comparar_distribuciones(y["test"], y_hat["test"])

    metadata = {
        "config": config,
        "demo": demo,
        "variables_entrada": {
            "numericas": d["numericas"],
            "logaritmicas": d["logaritmicas"],
            "categoricas": d["categoricas"],
        },
        "columnas_transformadas": list(preprocesador.get_feature_names_out()),
        "valores_recortados": recortados,
        "parametros_entrenables": red.contar_parametros(),
        "mejor_epoca": red.mejor_epoca,
        "epocas_entrenadas": len(red.historia["perdida_entrenamiento"]),
        "segundos_entrenamiento": segundos,
        "metricas": resultados,
        "distribucion_test": distribucion_test.to_dict(),
    }
    predicciones = {"y_test": y["test"], "y_hat_test": y_hat["test"]}
    return red, preprocesador, predicciones, metadata


def guardar(carpeta, red, preprocesador, predicciones, metadata):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    red.guardar(carpeta / "red.npz")
    joblib.dump(preprocesador, carpeta / "preprocesador.joblib")
    np.savez(carpeta / "predicciones_test.npz", **predicciones)
    with open(carpeta / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def imprimir_resumen(metadata):
    print(f"\nParámetros entrenables: {metadata['parametros_entrenables']}")
    print(f"Mejor época: {metadata['mejor_epoca']} de {metadata['epocas_entrenadas']}"
          f"  ({metadata['segundos_entrenamiento']:.1f} s)")
    print(f"Valores recortados por partición: {metadata['valores_recortados']}")
    tabla = pd.DataFrame(metadata["metricas"]).T[["n", "mae", "rmse", "media_observada",
                                                  "media_predicha", "ks_distribuciones"]]
    tabla["n"] = tabla["n"].astype(int)
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nDistribución en test:")
    print(pd.DataFrame(metadata["distribucion_test"]).to_string(float_format=lambda v: f"{v:.4f}"))


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                        help="cambia un valor de la configuración, p. ej. red.perdida=entropia_cruzada")
    parser.add_argument("--demo", action="store_true", help="usa una muestra de los datos")
    parser.add_argument("--silencioso", action="store_true", help="no imprime cada época")
    args = parser.parse_args(argumentos)

    config = cargar_config(args.config, "lgd", args.set)
    if not args.silencioso:
        imprimir_config(config)
    red, preprocesador, predicciones, metadata = entrenar(copy.deepcopy(config), demo=args.demo,
                                                          verbose=not args.silencioso)
    guardar(config["salida"]["carpeta"], red, preprocesador, predicciones, metadata)
    imprimir_resumen(metadata)
    print(f"\nModelo guardado en {config['salida']['carpeta']}/")


if __name__ == "__main__":
    main()
