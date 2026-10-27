import numpy as np
import pytest
from src.perdida_esperada import calcular_perdida_esperada


def test_unidades_y_valores():
    np.testing.assert_allclose(calcular_perdida_esperada([.1,0,1],[.8,.5,1],[10000,2000,300]),[800,0,300])


@pytest.mark.parametrize('pd,lgd,ead', [([.1],[.5,.6],[100]),([1.1],[.5],[100]),([.1],[.5],[-1]),([.1],[float('nan')],[100]),([],[],[])])
def test_rechaza_inconsistencias(pd,lgd,ead):
    with pytest.raises(ValueError):
        calcular_perdida_esperada(pd,lgd,ead)


def test_inferencia_usa_ambos_modelos_y_mismas_filas(tmp_path):
    import joblib
    import pandas as pd
    from src.preprocesamiento import construir_preprocesador
    from src.red_neuronal import RedNeuronal
    from scripts.parte2_lgd.estimar_perdida import estimar
    datos = pd.DataFrame({'id':['a','b','c','d'], 'funded_amnt':[100,200,300,400], 'x':[1.,2.,3.,4.]})
    for tarea, perdida in [('pd','entropia_cruzada'),('lgd','ecm')]:
        carpeta = tmp_path / tarea
        carpeta.mkdir()
        pre = construir_preprocesador(['x'],[])
        X = pre.fit_transform(datos[['x']])
        red = RedNeuronal(1, capas_ocultas=(), perdida=perdida)
        red.entrenar(X,[0,0,1,1],epocas=2)
        red.guardar(carpeta/'red.npz')
        joblib.dump(pre,carpeta/'preprocesador.joblib')
    salida = estimar(datos,tmp_path/'pd',tmp_path/'lgd')
    assert salida.id.tolist()==datos.id.tolist()
    np.testing.assert_allclose(salida.perdida_esperada_proxy,salida.pd*salida.lgd_proxy*datos.funded_amnt)
    with pytest.raises(ValueError):
        estimar(pd.concat([datos,datos]),tmp_path/'pd',tmp_path/'lgd')
