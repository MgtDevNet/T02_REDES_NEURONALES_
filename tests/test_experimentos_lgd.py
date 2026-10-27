import json
import numpy as np
import pytest
from scripts.parte2_lgd.preparar_lgd import preparar
from tests.test_preparar_lgd import muestra
from src.experimentos_lgd import ejecutar


def test_comparacion_y_exportacion(tmp_path):
    ruta=tmp_path/'datos.parquet'
    preparar(muestra()).to_parquet(ruta,index=False)
    candidatos=[dict(capas_ocultas=(3,),activacion='tanh',l2=0.,dropout=0.,tasa=.01)]
    r=ejecutar(ruta,tmp_path/'salida',demo=True,epocas=2,semillas=[0,1,2],candidatos=candidatos)
    assert len(r['corridas'])==3
    assert r['manifiesto']['conteos']=={'train':6,'valid':2,'test':2}
    assert r['manifiesto']['configuraciones_ols']==1
    assert r['manifiesto']['candidato_elegido']==0
    assert np.isfinite(r['resumen'].to_numpy()).all()
    meta=json.loads((tmp_path/'salida/modelo_red/metadata.json').read_text())
    assert meta['semilla']==0 and meta['demo']
    with pytest.raises(ValueError):
        ejecutar(ruta,tmp_path/'otro',demo=False,semillas=[0,1])
