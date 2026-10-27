"""Carga ambos modelos de Manuel y predice pérdida por crédito sin reentrenar.

El archivo de entradas debe tener una fila por crédito, id único, funded_amnt
 y las predictoras con los tipos numéricos/categóricos usados al entrenar.
No recibe directamente loan.csv: requiere convertir term, emp_length y fechas.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.red_neuronal import RedNeuronal
from src.perdida_esperada import calcular_perdida_esperada


def estimar(datos, carpeta_pd, carpeta_lgd, id_col='id'):
    if id_col not in datos or datos[id_col].isna().any() or datos[id_col].duplicated().any():
        raise ValueError('Se requiere un identificador presente y único por crédito')
    if 'funded_amnt' not in datos:
        raise ValueError('Falta funded_amnt para EAD proxy')
    predicciones = []
    for tarea, carpeta in [('pd', carpeta_pd), ('lgd', carpeta_lgd)]:
        carpeta = Path(carpeta)
        red = RedNeuronal.cargar(carpeta / 'red.npz')
        if tarea == 'pd' and (red.config['salida'] != 'sigmoide' or red.config['perdida'] != 'entropia_cruzada'):
            raise ValueError('PD requiere el modelo binario de incumplimiento')
        # Cargue únicamente artefactos joblib propios/de procedencia conocida.
        pre = joblib.load(carpeta / 'preprocesador.joblib')
        columnas = list(pre.feature_names_in_)
        faltan = set(columnas) - set(datos.columns)
        if faltan:
            raise ValueError(f'Faltan entradas para {tarea}: {sorted(faltan)}')
        X = pre.transform(datos[columnas])
        if not np.isfinite(X).all():
            raise ValueError(f'Entradas no finitas después del preprocesamiento de {tarea}')
        predicciones.append(red.predecir(X))
    pd_est, lgd_est = predicciones
    ead = datos.funded_amnt.to_numpy(dtype=float)
    perdida = calcular_perdida_esperada(pd_est, lgd_est, ead)
    return pd.DataFrame({id_col: datos[id_col].to_numpy(), 'pd': pd_est, 'lgd_proxy': lgd_est,
                         'ead_proxy': ead, 'perdida_esperada_proxy': perdida})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--creditos', required=True, help='CSV o parquet con predictores ya tipados, id y funded_amnt')
    p.add_argument('--modelo-pd', default='models/pd')
    p.add_argument('--modelo-lgd', default='models/lgd')
    p.add_argument('--id', default='id')
    p.add_argument('--salida', default='reports/perdida_esperada.csv')
    args = p.parse_args()
    ruta = Path(args.creditos)
    datos = pd.read_parquet(ruta) if ruta.suffix == '.parquet' else pd.read_csv(ruta)
    resultado = estimar(datos, args.modelo_pd, args.modelo_lgd, args.id)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False)
    print(f'{len(resultado)} créditos; pérdida esperada proxy total: {resultado.perdida_esperada_proxy.sum():.2f}')
    print(f'Archivo: {salida}')


if __name__ == '__main__':
    main()
