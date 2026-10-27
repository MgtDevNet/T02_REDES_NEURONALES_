"""Prueba de punta a punta del script de entrenamiento con datos sintéticos que tienen
las mismas columnas que se esperan del script de limpieza."""

import json

import joblib
import numpy as np
import pandas as pd

from scripts.parte1_pd.entrenar_pd import main
from src.red_neuronal import RedNeuronal


def datos_sinteticos(n=3000, semilla=0):
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 35000, n),
        "term": rng.choice([36, 60], n),
        "emp_length": rng.integers(0, 11, n).astype(float),
        "annual_inc": rng.lognormal(11, 0.5, n),
        "dti": rng.uniform(0, 35, n),
        "delinq_2yrs": rng.poisson(0.3, n),
        "antiguedad_credito_meses": rng.uniform(12, 400, n),
        "inq_last_6mths": rng.poisson(1, n),
        "mths_since_last_delinq": np.where(rng.random(n) < 0.5, np.nan, rng.uniform(0, 100, n)),
        "mths_since_last_record": np.where(rng.random(n) < 0.9, np.nan, rng.uniform(0, 100, n)),
        "open_acc": rng.integers(1, 30, n),
        "pub_rec": rng.poisson(0.1, n),
        "revol_bal": rng.lognormal(9, 1, n),
        "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.integers(2, 60, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[0.5, 0.1, 0.395, 0.005]),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
    })
    # columnas como las deja scripts/parte1_pd/preparar_datos_pd.py (las de arriba se mantienen para la LGD)
    df["antig_credito_meses"] = df["antiguedad_credito_meses"].round()
    df["nunca_mora"] = df["mths_since_last_delinq"].isna().astype(int)
    df["emp_length_na"] = 0
    logit = -2 + 0.04 * df["dti"] + 0.5 * df["inq_last_6mths"] - 0.5 * np.log(df["annual_inc"] / 60000)
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["particion"] = rng.choice(["train", "valid", "test"], n, p=[0.6, 0.2, 0.2])
    return df


def preparar(tmp_path):
    """Argumentos para entrenar con config.ini apuntando a datos sintéticos."""
    ruta_datos = tmp_path / "pd_limpio.parquet"
    datos_sinteticos().to_parquet(ruta_datos)
    return ["--config", "config.ini", "--silencioso",
            "--set", f"datos.ruta={ruta_datos}",
            "--set", f"salida.carpeta={tmp_path / 'modelo'}",
            "--set", "entrenamiento.epocas=30"]


def test_entrenamiento_guarda_lo_que_necesita_la_app(tmp_path):
    main(preparar(tmp_path))

    carpeta = tmp_path / "modelo"
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["metricas"]["test"]["auc"] > 0.6

    # La app reconstruye la predicción solo con los archivos guardados
    preprocesador = joblib.load(carpeta / "preprocesador.joblib")
    red = RedNeuronal.cargar(carpeta / "red.npz")
    cliente = datos_sinteticos(n=1, semilla=99).drop(columns=["y", "particion"])
    p = red.predecir(preprocesador.transform(cliente))
    assert 0 < p[0] < 1
    assert len(np.load(carpeta / "puntajes_poblacion.npy")) == 3000


def test_cambios_desde_linea_de_comandos(tmp_path):
    main(preparar(tmp_path) + ["--set", "red.capas_ocultas=", "--set", "red.dropout=0.0"])
    metadata = json.loads((tmp_path / "modelo" / "metadata.json").read_text(encoding="utf-8"))
    # sin capas ocultas: un peso por columna transformada + 1 sesgo (regresión logística)
    assert metadata["parametros_entrenables"] == len(metadata["columnas_transformadas"]) + 1
