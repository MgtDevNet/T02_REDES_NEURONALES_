"""Pruebas de las piezas de búsqueda, semillas y evaluación de la Parte 1."""

import numpy as np
import pandas as pd

from scripts.parte1_pd.buscar_hiperparametros_pd import aplicar, como_set, muestrear_mlp
from scripts.parte1_pd.evaluar_pd import curva_roc
from scripts.parte1_pd.repetir_semillas_pd import resumir
from src import metricas
from src.configuracion import cargar_config


def test_cada_muestra_es_una_configuracion_valida():
    config = cargar_config("config.ini", "pd")
    rng = np.random.default_rng(0)
    for _ in range(20):
        cambios = muestrear_mlp(rng)
        nueva = aplicar(config, cambios)
        assert nueva["red"]["capas_ocultas"] == cambios["red.capas_ocultas"]
        assert 1e-6 <= nueva["red"]["l2"] <= 1e-2
        assert 0 <= nueva["red"]["dropout"] < 1
        # el comando impreso se puede volver a leer con --set
        cargar_config("config.ini", "pd", [s.removeprefix("--set ")
                                           for s in como_set(cambios).split(" --set ")])
    assert config["red"]["capas_ocultas"] == [64, 32]      # el original no se modifica


def test_resumen_de_semillas_brier_menor_es_mejor():
    tabla = pd.DataFrame({"auc_test": [0.60, 0.62, 0.61], "gini_test": [0.2, 0.24, 0.22],
                          "ks_test": [0.15, 0.17, 0.16], "brier_test": [0.12, 0.11, 0.13]})
    r = resumir(tabla, "test")
    assert r.loc["auc", "mejor"] == 0.62 and r.loc["auc", "peor"] == 0.60
    assert r.loc["brier", "mejor"] == 0.11 and r.loc["brier", "peor"] == 0.13


def test_area_bajo_la_curva_roc_coincide_con_el_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500) + 0.3 * y
    fpr, tpr = curva_roc(y, p)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert abs(np.trapezoid(tpr, fpr) - metricas.auc(y, p)) < 1e-9
