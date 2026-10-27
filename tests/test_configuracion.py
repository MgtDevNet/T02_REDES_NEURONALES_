"""Pruebas de la lectura de config.ini."""

import json
import shutil

import pytest

from scripts.parte1_pd.entrenar_pd import main
from src.configuracion import cargar_config
from tests.test_entrenar_pd import datos_sinteticos


def test_config_ini_del_repositorio():
    pd_config = cargar_config("config.ini", "pd")
    assert pd_config["semilla"] == 42
    assert pd_config["red"]["capas_ocultas"] == [64, 32]
    assert pd_config["red"]["l2"] == pytest.approx(0.0014)
    assert pd_config["entrenamiento"]["optimizador"] == "adam"
    assert pd_config["scorecard"] == {"puntaje_base": 600, "odds_base": 50, "pdo": 20}
    assert len(pd_config["datos"]["numericas"]) == 13   # lista en varias líneas
    assert "antig_credito_meses" in pd_config["datos"]["numericas"]
    assert pd_config["datos"]["indicadoras"] == ["nunca_mora", "emp_length_na"]
    assert len(pd_config["datos"]["con_tope"]) == 7     # lista con comentario en la 1.ª línea

    lgd_config = cargar_config("config.ini", "lgd")
    assert lgd_config["red"]["perdida"] == "ecm"
    assert lgd_config["datos"]["recortar_objetivo"] == [0, 1]
    assert "scorecard" not in lgd_config


def test_cambios_con_set():
    config = cargar_config("config.ini", "pd", ["red.capas_ocultas=64,32,8", "red.capas_ocultas=",
                                                "entrenamiento.tasa=0.05", "semilla=7"])
    assert config["red"]["capas_ocultas"] == []
    assert config["entrenamiento"]["tasa"] == 0.05
    assert config["semilla"] == 7
    with pytest.raises(KeyError):
        cargar_config("config.ini", "pd", ["red.neuronas=10"])   # clave mal escrita


def test_editar_el_ini_cambia_el_modelo(tmp_path):
    # Simula la modificación en vivo: se edita el archivo y se vuelve a ejecutar
    ruta_datos = tmp_path / "pd_limpio.parquet"
    datos_sinteticos().to_parquet(ruta_datos)
    ruta_ini = tmp_path / "config.ini"
    shutil.copy("config.ini", ruta_ini)
    texto = ruta_ini.read_text(encoding="utf-8")
    texto = texto.replace("capas_ocultas = 64, 32          # neuronas por capa oculta",
                          "capas_ocultas = 5          # neuronas por capa oculta")
    texto = texto.replace("ruta = data/processed/pd_limpio.parquet", f"ruta = {ruta_datos}")
    texto = texto.replace("carpeta = models/pd ", f"carpeta = {tmp_path / 'modelo'} ")
    texto = texto.replace("epocas = 200                    # máximo de épocas",
                          "epocas = 5                    # máximo de épocas")
    ruta_ini.write_text(texto, encoding="utf-8")

    main(["--config", str(ruta_ini), "--silencioso"])
    metadata = json.loads((tmp_path / "modelo" / "metadata.json").read_text(encoding="utf-8"))
    n_columnas = len(metadata["columnas_transformadas"])
    assert metadata["config"]["red"]["capas_ocultas"] == [5]
    assert metadata["parametros_entrenables"] == (n_columnas * 5 + 5) + (5 + 1)
