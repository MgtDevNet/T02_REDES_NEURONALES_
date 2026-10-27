"""Experimentos reproducibles del punto 2; reutiliza la red de Manuel."""
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.metricas import mae, resumen_regresion, comparar_distribuciones
from src.preprocesamiento import construir_preprocesador
from src.red_neuronal import RedNeuronal
from src.configuracion import cargar_config
from scripts.parte2_lgd.entrenar_lgd import guardar


def ejecutar(ruta='data/processed/lgd_limpio.parquet', salida='reports/experimentos_lgd',
             demo=False, epocas=200, semillas=None, candidatos=None, config='config.ini'):
    """Selecciona solo con validación; test se usa al evaluar la configuración final.

    Una referencia OLS; búsqueda finita de redes con semilla 42; réplicas finales
    con semillas predefinidas. Guarda como modelo de despliegue la primera semilla,
    nunca la que minimiza el error de prueba. No cambia las particiones entre semillas.
    """
    inicio_total = time.perf_counter()
    semillas = list(range(3 if demo else 30)) if semillas is None else list(semillas)
    if not semillas or len(set(semillas)) != len(semillas) or (not demo and len(semillas) < 30):
        raise ValueError('Use semillas distintas; el modo completo requiere al menos 30')
    if epocas < 1:
        raise ValueError('epocas debe ser positiva')
    candidatos = candidatos if candidatos is not None else [
        dict(capas_ocultas=(16,), activacion='tanh', l2=0.0001, dropout=0., tasa=.001),
        dict(capas_ocultas=(32,16), activacion='relu', l2=0.0001, dropout=.1, tasa=.001),
        dict(capas_ocultas=(32,16), activacion='tanh', l2=.001, dropout=.1, tasa=.003),
    ]
    if not candidatos:
        raise ValueError('Declare al menos un candidato')
    cfg = cargar_config(config, 'lgd')
    cfg['datos']['ruta'] = str(ruta)
    df = pd.read_parquet(ruta)
    d = cfg['datos']
    if set(df[d['particion']].unique()) != {'train','valid','test'}:
        raise ValueError('Se requieren exactamente train, valid y test')
    if demo:
        df = df.groupby(d['particion'], group_keys=False).sample(frac=.2, random_state=42)
    columnas = list(dict.fromkeys(d['numericas'] + d['categoricas']))
    partes = {p: df.loc[df[d['particion']] == p] for p in ('train','valid','test')}
    y = {p: f[d['objetivo']].to_numpy(dtype=float) for p,f in partes.items()}
    if any(not len(v) or not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in y.values()):
        raise ValueError('LGD debe ser finita y estar en [0,1]; no se recorta el objetivo')
    pre = construir_preprocesador(d['numericas'],d['categoricas'],d['logaritmicas'])
    X = {'train':pre.fit_transform(partes['train'][columnas])}
    X.update({p:pre.transform(partes[p][columnas]) for p in ('valid','test')})
    if any(not np.isfinite(v).all() for v in X.values()):
        raise ValueError('Revisar predictoras transformadas no finitas')
    out = Path(salida)
    out.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()
    lineal = LinearRegression().fit(X['train'],y['train'])
    tiempo_ols = time.perf_counter()-inicio
    pred_ols = lineal.predict(X['test'])
    metricas_ols = resumen_regresion(y['test'],pred_ols)
    metricas_ols.update(parametros=X['train'].shape[1]+1,segundos=tiempo_ols)
    joblib.dump(lineal,out/'lineal.joblib')
    joblib.dump(pre,out/'preprocesador.joblib')
    pd.DataFrame([metricas_ols]).to_csv(out/'referencia.csv',index=False)

    def ajustar(candidato, semilla):
        candidato = copy.deepcopy(candidato)
        tasa = candidato.pop('tasa')
        red = RedNeuronal(X['train'].shape[1], salida='sigmoide',perdida='ecm',
                          semilla=semilla, **candidato)
        inicio = time.perf_counter()
        red.entrenar(X['train'],y['train'],X['valid'],y['valid'],optimizador='adam',
                     tasa=tasa,epocas=epocas,tam_lote=cfg['entrenamiento']['tam_lote'],
                     paciencia=cfg['entrenamiento']['paciencia'])
        return red,time.perf_counter()-inicio

    busqueda = []
    for i,candidato in enumerate(candidatos):
        red,segundos = ajustar(candidato,42)
        busqueda.append(dict(candidato=i,mae_valid=mae(y['valid'],red.predecir(X['valid'])),
                             segundos=segundos,parametros=red.contar_parametros()))
    tabla_busqueda = pd.DataFrame(busqueda)
    elegido = int(tabla_busqueda.loc[tabla_busqueda.mae_valid.idxmin(),'candidato'])
    tabla_busqueda.to_csv(out/'busqueda_validacion.csv',index=False)
    corridas = []
    predicciones = {'y_test':y['test'],'pred_lineal':pred_ols}
    for j,semilla in enumerate(semillas):
        red,segundos = ajustar(candidatos[elegido],semilla)
        pred = red.predecir(X['test'])
        fila = resumen_regresion(y['test'],pred)
        fila.update(semilla=semilla,segundos=segundos,parametros=red.contar_parametros(),mejor_epoca=red.mejor_epoca)
        corridas.append(fila)
        predicciones[f'pred_semilla_{semilla}'] = pred
        if j == 0:
            # Se exporta la primera semilla, declarada antes de ver prueba.
            resultados = {p:resumen_regresion(y[p],red.predecir(X[p])) for p in partes}
            metadata = dict(config=cfg,demo=demo,semilla=semilla,candidato=candidatos[elegido],
                            metricas=resultados,mejor_epoca=red.mejor_epoca,
                            parametros_entrenables=red.contar_parametros(),
                            distribucion_test=comparar_distribuciones(y['test'],pred).to_dict())
            guardar(out/'modelo_red',red,pre,{'y_test':y['test'],'y_hat_test':pred},metadata)
            np.savez(out/'historia_red.npz',**red.historia)
    corridas = pd.DataFrame(corridas)
    corridas.to_csv(out/'corridas.csv',index=False)
    resumen = corridas[['mae','rmse','ks_distribuciones']].agg(['mean','std','min','max']).rename(
        index={'mean':'media','std':'desviacion_estandar','min':'mejor','max':'peor'})
    resumen.to_csv(out/'resumen_semillas.csv',index_label='estadistico')
    np.savez_compressed(out/'predicciones_comparadas.npz',**predicciones)
    manifiesto = dict(demo=demo,semillas=semillas,epocas_maximas=epocas,candidatos=candidatos,
                      candidato_elegido=elegido,criterio='menor MAE de validacion; semilla busqueda 42',
                      configuraciones_ols=1,configuraciones_red=len(candidatos),
                      ajustes_finales_red=len(semillas),
                      segundos_ols=tiempo_ols,segundos_busqueda=float(tabla_busqueda.segundos.sum()),
                      segundos_replicas=float(corridas.segundos.sum()),
                      segundos_total=time.perf_counter()-inicio_total,
                      conteos={p:len(v) for p,v in y.items()},
                      columnas_transformadas=list(pre.get_feature_names_out()),
                      nota='Particiones aleatorias fijas; presupuesto distinto OLS/red declarado; no son pruebas temporales')
    (out/'experimento.json').write_text(json.dumps(manifiesto,indent=2),encoding='utf-8')
    return dict(salida=out,referencia=pd.DataFrame([metricas_ols]),busqueda=tabla_busqueda,
                corridas=corridas,resumen=resumen,manifiesto=manifiesto)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos',default='data/processed/lgd_limpio.parquet')
    parser.add_argument('--salida',default='reports/experimentos_lgd')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--epocas',type=int,default=200)
    args=parser.parse_args()
    resultado=ejecutar(args.datos,args.salida,args.demo,args.epocas)
    print(resultado['resumen'])
