"""Pruebas de las métricas de regresión y del script de entrenamiento de la LGD."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from scripts.parte2_lgd.entrenar_lgd import main
from src import metricas
from src.red_neuronal import RedNeuronal
from tests.test_entrenar_pd import datos_sinteticos


def test_mae_rmse_igual_a_sklearn():
    rng = np.random.default_rng(0)
    y, y_hat = rng.random(500), rng.random(500)
    assert metricas.mae(y, y_hat) == pytest.approx(mean_absolute_error(y, y_hat))
    assert metricas.rmse(y, y_hat) == pytest.approx(root_mean_squared_error(y, y_hat))


def test_ks_dos_muestras():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    assert metricas.ks_dos_muestras(a, a) == 0
    assert metricas.ks_dos_muestras(a, a + 10) == 1
    assert metricas.ks_dos_muestras(a, np.array([0.1, 0.2, 5.0, 6.0])) == pytest.approx(0.5)


def test_sin_capas_ocultas_es_regresion_lineal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 3))
    y = X @ np.array([0.5, -1.0, 2.0]) + 0.3 + rng.normal(0, 0.1, 400)
    red = RedNeuronal(3, capas_ocultas=(), salida="identidad", perdida="ecm", semilla=0)
    red.entrenar(X, y, tasa=0.05, epocas=2000, tam_lote=len(X))
    lineal = LinearRegression().fit(X, y)
    np.testing.assert_allclose(red.W[0][:, 0], lineal.coef_, atol=1e-3)
    assert red.b[0][0, 0] == pytest.approx(lineal.intercept_, abs=1e-3)


def preparar(tmp_path):
    df = datos_sinteticos()
    rng = np.random.default_rng(5)
    # LGD sintética que depende de las variables, con algunos valores fuera de [0, 1]
    df["lgd"] = 0.7 - 0.1 * np.log(df["annual_inc"] / 60000) + 0.1 * (df["term"] == 60) \
        + rng.normal(0, 0.15, len(df))
    ruta_datos = tmp_path / "lgd_limpio.parquet"
    df.drop(columns="y").to_parquet(ruta_datos)
    return ["--config", "config.ini", "--silencioso",
            "--set", f"datos.ruta={ruta_datos}",
            "--set", f"salida.carpeta={tmp_path / 'modelo'}",
            "--set", "entrenamiento.epocas=30"]


@pytest.mark.parametrize("perdida", ["ecm", "entropia_cruzada"])
def test_entrenamiento_lgd(tmp_path, perdida):
    main(preparar(tmp_path) + ["--set", f"red.perdida={perdida}"])

    carpeta = tmp_path / "modelo"
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    assert sum(metadata["valores_recortados"].values()) > 0
    predicciones = np.load(carpeta / "predicciones_test.npz")
    assert np.all((predicciones["y_hat_test"] > 0) & (predicciones["y_hat_test"] < 1))
    # mejor que predecir siempre la media de train
    assert metadata["metricas"]["test"]["rmse"] < np.std(predicciones["y_test"])


def test_regresion_lineal_desde_linea_de_comandos(tmp_path):
    main(preparar(tmp_path) + ["--set", "red.capas_ocultas=", "--set", "red.salida=identidad",
                               "--set", "red.dropout=0"])
    metadata = json.loads((tmp_path / "modelo" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parametros_entrenables"] == len(metadata["columnas_transformadas"]) + 1
