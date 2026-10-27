"""Preparación temporal de PD auxiliar y cohorte común para pérdida esperada."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.parte1_pd.preparar_datos_pd import preparar as preparar_pd
from src.configuracion import cargar_config


def preparar_pd_y_cohorte(bruto, salida):
    """Misma población y partición que el modelo PD del punto 1 (scripts/parte1_pd/preparar_datos_pd.py):
    solo créditos a 36 meses; train jun-2007 a dic-2011, valid ene-jun 2012,
    test jul-dic 2012. Así la cohorte entra directo a models/pd.

    La cohorte común es test de PD. Puede solaparse con train de LGD, por lo que
    el producto no se presenta como evaluación independiente de pérdida esperada.
    """
    estados=bruto.loan_status.astype('string').str.strip()
    if estados.isna().any():
        raise ValueError('loan_status ausente')
    # codificación del enunciado, plazo, partición y limpieza fija (falla si hay estados desconocidos)
    df=preparar_pd(bruto)
    cfg=cargar_config('config.ini','pd')['datos']
    columnas=list(dict.fromkeys(cfg['numericas']+cfg['indicadoras']+cfg['categoricas']))
    for nombre in ['train','valid','test']:
        parte=df.loc[df.particion==nombre]
        if parte.empty or set(parte.y.unique())!={0,1}:
            raise ValueError(f'PD auxiliar requiere ambas clases en la partición temporal {nombre}')
    destino=Path(salida);destino.mkdir(parents=True,exist_ok=True)
    ruta=destino/'pd_temporal.parquet'
    df[columnas+['y','particion']].to_parquet(ruta,index=False)
    if 'id' not in df:
        raise ValueError('Falta id original para alinear créditos')
    cohorte=df.loc[df.particion=='test',columnas+['id','funded_amnt']].copy()
    if cohorte.id.isna().any() or cohorte.id.duplicated().any():
        raise ValueError('IDs ausentes o duplicados; revisar antes de combinar modelos')
    montos=pd.to_numeric(cohorte.funded_amnt,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(montos).all() or (montos<=0).any():
        raise ValueError('Exposiciones no válidas en la cohorte')
    ruta_cohorte=destino/'cohorte_perdida.parquet'
    cohorte.to_parquet(ruta_cohorte,index=False)
    return ruta,ruta_cohorte,df.groupby('particion').y.agg(['count','mean'])
