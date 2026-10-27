"""Construye el parquet que consume la red de Manuel para el punto 2.

Ejemplo: python -m scripts.parte2_lgd.preparar_lgd --csv datos/loan/loan.csv
No descarga datos ni cambia la implementación de la red neuronal.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERICAS = [
    'loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
    'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
]
COLUMNAS = list(dict.fromkeys(NUMERICAS + [
    'loan_status', 'funded_amnt', 'total_rec_prncp', 'term', 'emp_length',
    'issue_d', 'earliest_cr_line', 'home_ownership', 'purpose',
]))


def transformar_predictoras(df):
    """Transformaciones deterministas compartidas; no aprende imputación ni escalado."""
    datos = df.copy()
    for col in NUMERICAS:
        datos[col] = pd.to_numeric(datos[col].astype('string').str.strip().str.rstrip('%'), errors='raise').astype(float)
    datos['term'] = pd.to_numeric(datos.term.astype('string').str.strip().str.replace(' months', '', regex=False), errors='raise')
    empleo = datos.emp_length.astype('string').str.strip().replace({'< 1 year': '0', '10+ years': '10', 'n/a': pd.NA})
    empleo = empleo.str.replace(r' years?$', '', regex=True)
    datos['emp_length'] = pd.to_numeric(empleo, errors='raise').astype(float)
    fecha = pd.to_datetime(datos.issue_d, format='%b-%Y', errors='raise')
    primera = pd.to_datetime(datos.earliest_cr_line, format='%b-%Y', errors='raise')
    datos['antiguedad_credito_meses'] = ((fecha.dt.year - primera.dt.year) * 12 + fecha.dt.month - primera.dt.month).astype(float)
    if (datos.antiguedad_credito_meses.dropna() < 0).any():
        raise ValueError('earliest_cr_line posterior a issue_d')
    # El log1p del preprocesador de Manuel requiere estas magnitudes no negativas.
    if (datos[['annual_inc', 'revol_bal']] < 0).any().any():
        raise ValueError('annual_inc y revol_bal deben ser no negativas')
    entradas = NUMERICAS + ['term', 'emp_length', 'antiguedad_credito_meses']
    if np.isinf(datos[entradas].to_numpy(dtype=float)).any():
        raise ValueError('Predictoras infinitas: revisar datos')
    for col in ['home_ownership', 'purpose']:
        datos[col] = datos[col].where(datos[col].notna(), np.nan).astype(object)
    return datos


def preparar(df, semilla=42):
    """LGD proxy y particiones 60/20/20, sin aprender estadísticas globales.

    La muestra Charged Off/Default reproduce el alcance del notebook de LGD;
    no redefine la clase positiva más amplia exigida para PD.
    """
    faltan = set(COLUMNAS) - set(df.columns)
    if faltan:
        raise ValueError(f'Columnas ausentes: {sorted(faltan)}')
    datos = df.loc[df.loan_status.astype('string').str.strip().isin(['Charged Off', 'Default'])].copy()
    if len(datos) < 10:
        raise ValueError('Se requieren al menos 10 créditos en la muestra LGD')
    for col in NUMERICAS + ['funded_amnt', 'total_rec_prncp']:
        # Permite porcentajes como "54.2%"; se conservan unidades 0..100.
        datos[col] = pd.to_numeric(datos[col].astype('string').str.strip().str.rstrip('%'), errors='raise').astype(float)
    campos = datos[['funded_amnt', 'total_rec_prncp']].to_numpy()
    if not np.isfinite(campos).all() or (datos.funded_amnt <= 0).any() or (datos.total_rec_prncp < 0).any():
        raise ValueError('Revisar funded_amnt y total_rec_prncp: finitos, monto > 0 y capital recibido >= 0')
    datos['lgd'] = 1 - datos.total_rec_prncp / datos.funded_amnt
    if not datos.lgd.between(0, 1).all():
        raise ValueError('LGD proxy fuera de [0,1]; revisar los registros, sin recorte automático')
    datos = transformar_predictoras(datos)
    entradas = NUMERICAS + ['term', 'emp_length', 'antiguedad_credito_meses']
    datos = datos.reset_index(drop=True)
    desarrollo, prueba = train_test_split(datos.index, test_size=0.2, random_state=semilla)
    entrenamiento, validacion = train_test_split(desarrollo, test_size=0.25, random_state=semilla)
    datos['particion'] = ''
    for nombre, indices in [('train', entrenamiento), ('valid', validacion), ('test', prueba)]:
        datos.loc[indices, 'particion'] = nombre
    # Solo predictores ex ante y objetivo: nunca capital recibido como entrada.
    salida = datos[entradas + ['home_ownership', 'purpose', 'lgd', 'particion']].copy()
    for col in entradas:
        if salida.loc[salida.particion == 'train', col].isna().all():
            raise ValueError(f'Predictora sin valores observados en entrenamiento: {col}')
    return salida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--salida', default='data/processed/lgd_limpio.parquet')
    parser.add_argument('--semilla', type=int, default=42)
    args = parser.parse_args()
    datos = preparar(pd.read_csv(args.csv, usecols=COLUMNAS, low_memory=False), args.semilla)
    ruta = Path(args.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos.to_parquet(ruta, index=False)
    metadata = {'formula': '1 - total_rec_prncp / funded_amnt',
                'poblacion': ['Charged Off', 'Default'], 'semilla_particion': args.semilla,
                'particion': 'aleatoria 60/20/20; no es validacion temporal',
                'conteos': datos.particion.value_counts().to_dict(), 'origen': str(Path(args.csv))}
    ruta.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
