"""Pruebas de la aplicación web: la lógica (app/motor.py) y que las páginas corran sin errores."""

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.motor import CAMPOS, RAIZ, Modelo, banda, campos_del_modelo, leer_config_app, solicitante_por_defecto


@pytest.fixture(scope="module")
def logistica():
    return Modelo(leer_config_app()["modelo"])


def test_evaluar_y_puntos_suman_en_la_logistica(logistica):
    s = solicitante_por_defecto()
    r = logistica.evaluar(s).iloc[0]
    assert 0 < r["pd"] < 1 and 0 <= r["percentil"] <= 100
    assert r["puntaje"] == pytest.approx(logistica.scorecard.puntaje(r["pd"]))
    base, puntos, interaccion = logistica.puntos(s)
    assert base + puntos.sum() == pytest.approx(r["puntaje"], abs=1e-6)
    assert abs(interaccion) < 1e-6


def test_mas_consultas_bajan_el_puntaje_y_mora_vacia_es_nunca(logistica):
    s = solicitante_por_defecto()
    puntajes = logistica.barrido(s, "inq_last_6mths", [0, 2, 5])
    assert puntajes[0] > puntajes[1] > puntajes[2]
    filas = logistica.filas([{**s, "mths_since_last_delinq": None}, {**s, "mths_since_last_delinq": 12}])
    assert filas["nunca_mora"].tolist() == [1, 0]


def test_los_campos_no_salen_del_rango_de_entrenamiento(logistica):
    rangos = logistica.metadata["rangos"]
    for var, c in CAMPOS.items():
        if c.opciones or var not in rangos:
            continue
        minimo, maximo = (c.minimo * 12, c.maximo * 12) if var == "antig_credito_meses" else (c.minimo, c.maximo)
        assert maximo <= rangos[var][1], var
    assert CAMPOS["dti"].maximo < 30


def test_bandas_cubren_todos_los_percentiles():
    assert [banda(p)[1] for p in (0, 25, 50, 70, 100)] == \
        ["Riesgo alto", "Por mejorar", "Promedio", "Bueno", "Excelente"]


@pytest.mark.parametrize("pagina", ["app/evaluar.py", "app/como_funciona.py"])
def test_paginas_corren_sin_errores(pagina):
    at = AppTest.from_file(str(RAIZ / pagina), default_timeout=60).run()
    assert not at.exception


def test_simulador_y_cambio_de_modelo():
    at = AppTest.from_file(str(RAIZ / "app/evaluar.py"), default_timeout=60).run()
    at.selectbox(key="var_simulada").set_value("purpose").run()
    at.selectbox(key="nuevo_purpose").set_value("small_business").run()
    assert float(at.metric[0].delta.split()[0]) < 0          # pequeño negocio resta puntos
    at.radio(key="modelo_elegido").set_value("Red neuronal (comparación)").run()
    assert not at.exception
    assert np.isfinite(float(at.metric[0].value))


def test_controles_llegan_justo_al_tope_del_modelo(logistica):
    campos = campos_del_modelo(logistica)
    for var, tope in logistica.preprocesador.topes_.items():
        c = campos[var]
        escala = 12 if var == "antig_credito_meses" else 1
        assert tope <= c.maximo * escala < tope + c.paso * escala, var    # alcanza el tope, sin pasarse
        assert c.minimo <= c.defecto <= c.maximo
    assert campos["delinq_2yrs"].opciones == {0: "0", 1: "1", 2: "2 o más"}
    # por encima del tope el puntaje ya no cambia: por eso no se ofrece
    s = solicitante_por_defecto()
    assert np.ptp(logistica.barrido(s, "delinq_2yrs", [2, 5, 10])) == pytest.approx(0)


def test_simular_moras_con_lista():
    at = AppTest.from_file(str(RAIZ / "app/evaluar.py"), default_timeout=60).run()
    at.selectbox(key="var_simulada").set_value("delinq_2yrs").run()
    at.selectbox(key="nuevo_delinq_2yrs").set_value(2).run()
    assert not at.exception
    assert float(at.metric[0].delta.split()[0]) < 0
