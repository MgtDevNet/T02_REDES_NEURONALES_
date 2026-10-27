"""
Entrena el modelo de probabilidad de incumplimiento con un solo comando.

    python -m scripts.parte1_pd.entrenar_pd                                     # usa config.ini
    python -m scripts.parte1_pd.entrenar_pd --demo                              # muestra de los datos (< 2 min)
    python -m scripts.parte1_pd.entrenar_pd --set red.capas_ocultas=64,32 --set red.dropout=0.3
    python -m scripts.parte1_pd.entrenar_pd --set red.capas_ocultas= --set salida.carpeta=models/pd_logistica

Los hiperparámetros se cambian en config.ini (secciones [pd.xxx]) o con --set.

Guarda en la carpeta de salida todo lo que la aplicación web necesita para no reentrenar:
    red.npz                  pesos y arquitectura de la red
    preprocesador.joblib     imputación, escalado y one-hot ajustados con train
    puntajes_poblacion.npy   puntajes de los créditos etiquetados, para calcular percentiles
    metadata.json            variables, configuración, scorecard, métricas y tiempos
"""

import argparse
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import metricas
from src.configuracion import cargar_config, imprimir_config
from src.preprocesamiento import PreprocesadorCredito
from src.red_neuronal import RedNeuronal
from src.scorecard import Scorecard

PARTICIONES = ("train", "valid", "test")


def cargar_datos(config, demo):
    d = config["datos"]
    df = pd.read_parquet(d["ruta"])
    if demo:
        df = (df.groupby(d["particion"], group_keys=False)
                .sample(frac=d["fraccion_demo"], random_state=config["semilla"]))
    variables = list(dict.fromkeys(d["numericas"] + d.get("indicadoras", []) + d["categoricas"]))
    return {
        nombre: (parte[variables], parte[d["objetivo"]].to_numpy(dtype=float))
        for nombre, parte in df.groupby(d["particion"])
        if nombre in PARTICIONES
    }


def entrenar(config, demo=False, verbose=True):
    inicio = time.perf_counter()
    d, r, e = config["datos"], config["red"], config["entrenamiento"]

    datos = cargar_datos(config, demo)
    faltantes = set(PARTICIONES) - set(datos)
    if faltantes:
        raise ValueError(f"Faltan particiones en los datos: {sorted(faltantes)}")

    preprocesador = PreprocesadorCredito(
        d["numericas"], d["categoricas"], logaritmicas=d["logaritmicas"],
        con_tope=d["con_tope"], indicadoras=d["indicadoras"],
        frecuencia_minima=d["frecuencia_minima"])
    X = {"train": preprocesador.fit_transform(datos["train"][0])}
    X.update({nombre: preprocesador.transform(datos[nombre][0]) for nombre in ("valid", "test")})
    X_train = X["train"]
    y = {nombre: datos[nombre][1] for nombre in PARTICIONES}

    red = RedNeuronal(X_train.shape[1], capas_ocultas=r["capas_ocultas"],
                      activacion=r["activacion"], l2=r["l2"], dropout=r["dropout"],
                      semilla=config["semilla"])
    red.entrenar(X["train"], y["train"], X["valid"], y["valid"],
                 optimizador=e["optimizador"], tasa=e["tasa"], epocas=e["epocas"],
                 tam_lote=e["tam_lote"], paciencia=e["paciencia"], verbose=verbose)
    segundos = time.perf_counter() - inicio

    p = {nombre: red.predecir(X[nombre]) for nombre in PARTICIONES}
    resultados = {nombre: metricas.resumen(y[nombre], p[nombre]) for nombre in PARTICIONES}

    scorecard = Scorecard(**config["scorecard"])
    puntajes = scorecard.puntaje(np.concatenate([p[nombre] for nombre in PARTICIONES]))

    metadata = {
        "config": config,
        "demo": demo,
        "variables_entrada": {
            "numericas": d["numericas"],
            "logaritmicas": d["logaritmicas"],
            "con_tope": d["con_tope"],
            "indicadoras": d["indicadoras"],
            "categoricas": d["categoricas"],
        },
        "columnas_transformadas": list(preprocesador.get_feature_names_out()),
        # solicitante de referencia para los puntos por variable (src/scorecard.py)
        "referencia_transformada": X_train.mean(axis=0).tolist(),
        # categorías que reconoce el modelo (las raras de train quedan en "otras")
        "categorias": preprocesador.categorias_,
        # la app limita cada campo a este rango para que la red no extrapole
        # (p. ej. dti: el máximo en train es ~30, pero en 2012-S2 llega a 35)
        "rangos": {
            col: [float(datos["train"][0][col].min()), float(datos["train"][0][col].max())]
            for col in d["numericas"]
        },
        "scorecard": scorecard.parametros(),
        "parametros_entrenables": red.contar_parametros(),
        "mejor_epoca": red.mejor_epoca,
        "epocas_entrenadas": len(red.historia["perdida_entrenamiento"]),
        "segundos_entrenamiento": segundos,
        "metricas": resultados,
    }
    return red, preprocesador, puntajes, metadata


def guardar(carpeta, red, preprocesador, puntajes, metadata):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    red.guardar(carpeta / "red.npz")
    joblib.dump(preprocesador, carpeta / "preprocesador.joblib")
    np.save(carpeta / "puntajes_poblacion.npy", puntajes)
    with open(carpeta / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def imprimir_resumen(metadata):
    print(f"\nParámetros entrenables: {metadata['parametros_entrenables']}")
    print(f"Mejor época: {metadata['mejor_epoca']} de {metadata['epocas_entrenadas']}"
          f"  ({metadata['segundos_entrenamiento']:.1f} s)")
    tabla = pd.DataFrame(metadata["metricas"]).T[["n", "tasa_malos", "auc", "gini", "ks", "brier"]]
    tabla["n"] = tabla["n"].astype(int)
    print(tabla.to_string(float_format=lambda v: f"{v:.4f}"))


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                        help="cambia un valor de la configuración, p. ej. red.dropout=0.3")
    parser.add_argument("--demo", action="store_true", help="usa una muestra de los datos")
    parser.add_argument("--silencioso", action="store_true", help="no imprime cada época")
    args = parser.parse_args(argumentos)

    config = cargar_config(args.config, "pd", args.set)
    if not args.silencioso:
        imprimir_config(config)
    red, preprocesador, puntajes, metadata = entrenar(copy.deepcopy(config), demo=args.demo,
                                                      verbose=not args.silencioso)
    guardar(config["salida"]["carpeta"], red, preprocesador, puntajes, metadata)
    imprimir_resumen(metadata)
    print(f"\nModelo guardado en {config['salida']['carpeta']}/")


if __name__ == "__main__":
    main()
