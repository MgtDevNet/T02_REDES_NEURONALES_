"""Pruebas de la limpieza (scripts/parte1_pd/preparar_datos_pd.py) y del preprocesamiento de la Parte 1."""

import numpy as np
import pandas as pd
import pytest

from scripts.parte1_pd.preparar_datos_pd import preparar
from src.preprocesamiento import PreprocesadorCredito


def fila(**cambios):
    base = {"loan_status": "Fully Paid", "issue_d": "Mar-2011", "loan_amnt": 10000.0,
            "term": " 36 months", "emp_length": "5 years", "home_ownership": "RENT",
            "annual_inc": 50000.0, "purpose": "credit_card", "dti": 15.0, "delinq_2yrs": 0.0,
            "earliest_cr_line": "Mar-2001", "inq_last_6mths": 1.0,
            "mths_since_last_delinq": np.nan, "open_acc": 8.0, "pub_rec": 0.0,
            "revol_bal": 5000.0, "revol_util": 40.0, "total_acc": 20.0}
    base.update(cambios)
    return base


def test_objetivo_plazo_y_particion():
    crudo = pd.DataFrame([
        fila(),                                                     # train
        fila(loan_status="Charged Off", issue_d="Jun-2012"),        # valid
        fila(loan_status="Late (31-120 days)", issue_d="Jul-2012"), # test
        fila(loan_status="Current"),                                # NA: fuera
        fila(term=" 60 months"),                                    # 60 meses: fuera
        fila(issue_d="Jan-2013"),                                   # censurado: fuera
    ])
    d = preparar(crudo)
    assert d["particion"].tolist() == ["train", "valid", "test"]
    assert d["y"].tolist() == [0, 1, 1]


def test_estado_desconocido_detiene_la_limpieza():
    with pytest.raises(AssertionError):
        preparar(pd.DataFrame([fila(loan_status="Algo nuevo")]))


def test_reglas_fijas():
    crudo = pd.DataFrame([
        fila(mths_since_last_delinq=0.0, delinq_2yrs=0.0),   # cero-código -> nunca mora
        fila(mths_since_last_delinq=0.0, delinq_2yrs=1.0),   # cero genuino
        fila(emp_length="< 1 year", revol_util=np.nan, revol_bal=0.0, pub_rec=5.0),
        fila(emp_length="n/a", home_ownership="NONE"),
        fila(inq_last_6mths=np.nan),                         # sin historial: fuera
    ])
    d = preparar(crudo)
    assert len(d) == 4
    assert d["nunca_mora"].tolist() == [1, 0, 1, 1]
    assert np.isnan(d.loc[0, "mths_since_last_delinq"]) and d.loc[1, "mths_since_last_delinq"] == 0
    assert d.loc[2, "emp_length"] == 0 and d.loc[2, "revol_util"] == 0 and d.loc[2, "pub_rec"] == 2
    assert d.loc[3, "emp_length_na"] == 1 and d.loc[3, "home_ownership"] == "OTHER"
    assert (d["antig_credito_meses"] == 120).all()


def test_preprocesador_aprende_solo_de_train():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "ingreso": np.r_[rng.lognormal(11, 0.5, 999), [1e9]],   # un valor extremo
        "deuda": np.r_[rng.uniform(0, 30, 990), [np.nan] * 10],
        "flag": rng.integers(0, 2, 1000),
        "proposito": ["a"] * 600 + ["b"] * 395 + ["rara"] * 5,
    })
    pre = PreprocesadorCredito(["ingreso", "deuda"], ["proposito"], logaritmicas=["ingreso"],
                               con_tope=["ingreso"], indicadoras=["flag"])
    X = pre.fit_transform(train)

    assert list(pre.get_feature_names_out()) == ["ingreso", "deuda", "flag",
                                                 "proposito_b", "proposito_otras"]
    assert not np.isnan(X).any()
    assert X[:, 0].max() < 4                     # el extremo quedó recortado en el p99 antes del log
    assert np.allclose(X[:, :2].mean(axis=0), 0) and np.allclose(X[:, :2].std(axis=0), 1)

    nuevo = pd.DataFrame({"ingreso": [np.nan], "deuda": [100.0], "flag": [1], "proposito": ["nunca_vista"]})
    x = pre.transform(nuevo)[0]
    assert x[0] == pytest.approx((np.log1p(pre.medianas_["ingreso"]) - pre.medias_["ingreso"])
                                 / pre.desviaciones_["ingreso"])
    assert x[3:].tolist() == [0, 1]              # categoría no vista -> "otras"


def test_logaritmica_debe_ser_numerica():
    with pytest.raises(ValueError):
        PreprocesadorCredito(["a"], logaritmicas=["b"])
