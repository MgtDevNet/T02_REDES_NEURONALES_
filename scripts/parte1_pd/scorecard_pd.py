"""
Scorecard de la Parte 1: de la probabilidad de incumplimiento a puntos por variable.

    python -m scripts.parte1_pd.scorecard_pd                          # regresión logística (models/pd_logistica)
    python -m scripts.parte1_pd.scorecard_pd --modelo models/pd       # la red neuronal

Parámetros (config.ini, sección [pd.scorecard]): puntaje base, odds de referencia y PDO.
    factor = PDO / ln(2)
    offset = puntaje_base - factor * ln(odds_base)
    puntaje = offset + factor * ln((1 - p) / p) = offset - factor * logit(p)

Los puntos de cada variable son los que gana o pierde un cliente frente al solicitante
promedio de entrenamiento (ver puntos_por_variable en src/scorecard.py). En la regresión
logística suman exactamente el puntaje; en la red queda un residuo de interacciones.

Genera:
    reports/scorecard_pd.csv            puntos por variable en valores representativos
    reports/scorecard_pd_ejemplos.csv   descomposición del puntaje de cinco clientes de prueba
"""

import argparse

import numpy as np
import pandas as pd

from scripts.parte1_pd.evaluar_pd import cargar_modelo
from src.preprocesamiento import INDICADOR_DE
from src.scorecard import Scorecard, puntos_por_variable
from src.variables import NOMBRES

CUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def cliente_tipico(train, preprocesador):
    """Fila con la mediana de cada numérica y la moda de cada categórica e indicadora."""
    fila = {c: train[c].median() for c in preprocesador.numericas}
    fila.update({c: train[c].mode()[0] for c in preprocesador.indicadoras + preprocesador.categoricas})
    for indicador, variable in INDICADOR_DE.items():
        if fila.get(indicador) == 1:          # la moda es "sin dato" (p. ej. nunca ha estado en mora)
            fila[variable] = np.nan
    return pd.DataFrame([fila])


def valores_representativos(train, variable, preprocesador):
    if variable in preprocesador.categoricas:
        return list(preprocesador.categorias_[variable])
    valores = sorted(set(np.round(train[variable].quantile(CUANTILES).to_numpy(), 2)))
    if variable in INDICADOR_DE.values():
        valores = [np.nan] + valores          # "sin mora" / "no informó"
    return valores


def tabla_scorecard(modelo, scorecard, train):
    pre = modelo["preprocesador"]
    referencia = np.array(modelo["metadata"]["referencia_transformada"])
    grupos = pre.grupos()
    base = cliente_tipico(train, pre)
    indicador_de = {v: i for i, v in INDICADOR_DE.items()}

    filas = []
    for variable, columnas in grupos.items():
        if not columnas:
            continue
        for valor in valores_representativos(train, variable, pre):
            x = base.copy()
            x[variable] = valor
            if variable in indicador_de:
                x[indicador_de[variable]] = int(pd.isna(valor))
            _, puntos, _ = puntos_por_variable(modelo["red"], scorecard, pre.transform(x),
                                               referencia, {variable: columnas})
            etiqueta = ("sin dato" if pd.isna(valor) else valor)
            filas.append({"variable": variable, "nombre": NOMBRES.get(variable, variable),
                          "valor": etiqueta, "puntos": round(float(puntos[variable][0]), 1)})
    return pd.DataFrame(filas)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="models/pd_logistica")
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    args = parser.parse_args(argumentos)

    modelo = cargar_modelo(args.modelo)
    scorecard = Scorecard(**modelo["metadata"]["config"]["scorecard"])
    datos = pd.read_parquet(args.datos)
    train = datos[datos["particion"] == "train"]
    test = datos[datos["particion"] == "test"].reset_index(drop=True)
    pre = modelo["preprocesador"]
    referencia = np.array(modelo["metadata"]["referencia_transformada"])

    s = scorecard.parametros()
    print(f"Modelo: {modelo['nombre']}")
    print(f"Puntaje base {s['puntaje_base']} con odds {s['odds_base']}:1, PDO {s['pdo']}")
    print(f"factor = {s['factor']:.4f}   offset = {s['offset']:.4f}")

    tabla = tabla_scorecard(modelo, scorecard, train)
    tabla.to_csv("reports/scorecard_pd.csv", index=False)
    print("\nPuntos frente al solicitante promedio de entrenamiento:")
    print(tabla.drop(columns="variable").to_string(index=False))

    # Cinco clientes de prueba, del más riesgoso al menos riesgoso
    X = pre.transform(test)
    p = modelo["red"].predecir(X)
    elegidos = np.argsort(p)[np.linspace(len(p) - 1, 0, 5).astype(int)]
    base, puntos, interaccion = puntos_por_variable(modelo["red"], scorecard, X[elegidos],
                                                    referencia, pre.grupos())
    ejemplos = puntos.round(1)
    ejemplos.insert(0, "PD", p[elegidos].round(4))
    ejemplos.insert(1, "puntaje", scorecard.puntaje(p[elegidos]).round(1))
    ejemplos["interaccion"] = interaccion.round(2)
    ejemplos.to_csv("reports/scorecard_pd_ejemplos.csv", index=False)
    print(f"\nPuntaje del solicitante promedio (base): {base:.1f}")
    print("Cinco clientes de prueba: puntaje = base + suma de puntos + interacción")
    print(ejemplos.T.to_string())
    print(f"\nInteracción máxima en valor absoluto: {np.abs(interaccion).max():.4f} puntos"
          + ("  (logística: debe ser ~0)" if not modelo["metadata"]["config"]["red"]["capas_ocultas"] else ""))


if __name__ == "__main__":
    main()
