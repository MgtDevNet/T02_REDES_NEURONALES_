import numpy as np
import pandas as pd
from tests.test_preparar_lgd import muestra
from src.integracion_pd_lgd import preparar_pd_y_cohorte
from src.discusion_lgd import redactar
from src.experimentos_lgd import ejecutar
from scripts.parte2_lgd.preparar_lgd import preparar


def muestra_temporal():
    d=pd.concat([muestra()]*4,ignore_index=True)
    d['id']=np.arange(len(d))
    # una fecha por partición de la PD: train (<= dic-2011), valid (ene-jun 2012), test (jul-dic 2012)
    d['issue_d']=[['Jan-2010','Mar-2012','Sep-2012'][i%3] for i in range(len(d))]
    d['loan_status']=['Fully Paid' if i%2 else 'Charged Off' for i in range(len(d))]
    return d


def test_cohorte_y_pd_temporal(tmp_path):
    ruta,cohorte,resumen=preparar_pd_y_cohorte(muestra_temporal(),tmp_path)
    datos=pd.read_parquet(ruta)
    assert set(datos.particion)=={'train','valid','test'}
    for _,f in datos.groupby('particion'):assert set(f.y)=={0,1}
    c=pd.read_parquet(cohorte)
    assert c.id.is_unique and len(c)==resumen.loc['test','count']
    assert 'funded_amnt' in c and 'total_rec_prncp' not in c


def test_discusion_usa_resultados(tmp_path):
    ruta=tmp_path/'lgd.parquet'
    preparar(muestra()).to_parquet(ruta,index=False)
    ejecutar(ruta,tmp_path/'salida',demo=True,epocas=1,semillas=[0,1,2],
             candidatos=[dict(capas_ocultas=(2,),activacion='tanh',l2=0.,dropout=0.,tasa=.01)])
    texto=redactar(tmp_path/'salida')
    assert 'MAE' in texto and '3 corridas' in texto and 'Demostración preliminar' in texto
    assert (tmp_path/'salida/discusion_resultados.md').exists()
