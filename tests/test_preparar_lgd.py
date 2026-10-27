import numpy as np
import pandas as pd
import pytest
from scripts.parte2_lgd.preparar_lgd import COLUMNAS, preparar


def muestra():
    datos = pd.DataFrame({c: [1.0] * 50 for c in COLUMNAS})
    datos['loan_status'] = 'Charged Off'
    datos['funded_amnt'] = 1000
    datos['total_rec_prncp'] = 200
    datos['term'] = '36 months'
    datos['emp_length'] = '10+ years'
    datos['issue_d'] = 'Jan-2010'
    datos['earliest_cr_line'] = 'Jan-2000'
    datos['home_ownership'] = 'RENT'
    datos['purpose'] = 'other'
    return datos


def test_objetivo_transformaciones_y_particiones():
    a = preparar(muestra())
    pd.testing.assert_frame_equal(a, preparar(muestra()))
    assert np.allclose(a.lgd, .8)
    assert (a.antiguedad_credito_meses == 120).all()
    assert (a.emp_length == 10).all()
    assert a.particion.value_counts().to_dict() == {'train': 30, 'valid': 10, 'test': 10}
    assert not {'funded_amnt', 'total_rec_prncp', 'loan_status'} & set(a.columns)


@pytest.mark.parametrize('campo,valor', [('funded_amnt',0), ('total_rec_prncp',1100), ('annual_inc',-1)])
def test_no_oculta_errores(campo, valor):
    df = muestra()
    df.loc[0, campo] = valor
    with pytest.raises(ValueError):
        preparar(df)
