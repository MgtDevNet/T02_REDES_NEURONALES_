"""Pérdida esperada sobre capital original: PD * LGD_proxy * monto financiado."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def calcular_perdida_esperada(pd, lgd, ead):
    """Devuelve un vector monetario; rechaza desalineación y rangos inválidos.

    EAD debe ser una exposición determinista por crédito en la misma moneda.
    La igualdad producto no supone independencia PD/LGD: LGD es condicional
    al incumplimiento. Con EAD aleatoria se necesita modelar su relación con LGD.
    """
    pd, lgd, ead = (np.asarray(v, dtype=float) for v in (pd, lgd, ead))
    if pd.ndim != 1 or not len(pd) or pd.shape != lgd.shape or pd.shape != ead.shape:
        raise ValueError('PD, LGD y EAD deben ser vectores alineados no vacíos')
    if not all(np.isfinite(v).all() for v in (pd, lgd, ead)):
        raise ValueError('No se admiten valores ausentes o no finitos')
    if ((pd < 0) | (pd > 1)).any() or ((lgd < 0) | (lgd > 1)).any() or (ead < 0).any():
        raise ValueError('PD y LGD deben estar en [0,1]; EAD debe ser no negativa')
    return pd * lgd * ead


def estimar_perdida_esperada_cartera(ruta_modelo_pd='models/pd/red.npz',
                                     ruta_modelo_lgd='models/lgd/red.npz',
                                     tamano_muestra=10000,
                                     ruta_datos='data/processed/cohorte_perdida.parquet',
                                     semilla=42):
    """Estima la pérdida esperada agregada y por crédito sobre la cohorte de prueba.

    Utiliza el identificador real (`id`) y el capital financiado real (`funded_amnt`).
    Función de integración compatible con la Opción C del README.
    """
    from src.red_neuronal import RedNeuronal

    p_pd = Path(ruta_modelo_pd).parent
    p_lgd = Path(ruta_modelo_lgd).parent

    red_pd = RedNeuronal.cargar(Path(ruta_modelo_pd))
    pre_pd = joblib.load(p_pd / 'preprocesador.joblib')

    red_lgd = RedNeuronal.cargar(Path(ruta_modelo_lgd))
    pre_lgd = joblib.load(p_lgd / 'preprocesador.joblib')

    ruta = Path(ruta_datos)
    if not ruta.exists():
        ruta = Path('data/processed/pd_limpio.parquet')
    df = pd.read_parquet(ruta)

    if 'particion' in df.columns:
        df_eval = df[df['particion'] == 'test'].copy()
    else:
        df_eval = df.copy()

    if tamano_muestra is not None and len(df_eval) > tamano_muestra:
        df_eval = df_eval.sample(n=tamano_muestra, random_state=semilla)
    df_eval = df_eval.reset_index(drop=True)

    if 'id' in df_eval.columns:
        id_real = df_eval['id'].to_numpy()
    else:
        id_real = df_eval.index.to_numpy()

    if 'funded_amnt' in df_eval.columns:
        ead = pd.to_numeric(df_eval['funded_amnt'], errors='coerce').to_numpy(dtype=float)
    else:
        ead = pd.to_numeric(df_eval['loan_amnt'], errors='coerce').to_numpy(dtype=float)

    if 'antiguedad_credito_meses' not in df_eval.columns and 'antig_credito_meses' in df_eval.columns:
        df_eval['antiguedad_credito_meses'] = df_eval['antig_credito_meses']
    if 'term' not in df_eval.columns:
        df_eval['term'] = 36.0
    if 'mths_since_last_record' not in df_eval.columns:
        df_eval['mths_since_last_record'] = np.nan

    X_pd = pre_pd.transform(df_eval[list(pre_pd.feature_names_in_)])
    pd_hat = red_pd.predecir(X_pd)

    X_lgd = pre_lgd.transform(df_eval[list(pre_lgd.feature_names_in_)])
    lgd_hat = red_lgd.predecir(X_lgd)

    el = calcular_perdida_esperada(pd_hat, lgd_hat, ead)

    total_ead = float(ead.sum())
    total_el = float(el.sum())
    tasa_perdida = total_el / total_ead if total_ead > 0 else 0.0

    df_perdidas = pd.DataFrame({
        'id': id_real,
        'funded_amnt': ead,
        'pd': pd_hat,
        'lgd_proxy': lgd_hat,
        'ead_proxy': ead,
        'perdida_esperada_proxy': el
    })

    resumen_cartera = {
        'total_creditos': len(df_perdidas),
        'ead_total': total_ead,
        'perdida_esperada_total': total_el,
        'tasa_perdida_esperada': tasa_perdida,
        'pd_media': float(pd_hat.mean()),
        'lgd_media': float(lgd_hat.mean()),
        'el_media_por_credito': float(el.mean())
    }

    return resumen_cartera, df_perdidas
