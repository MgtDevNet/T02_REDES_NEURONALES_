"""Regenera métricas, tablas y figuras desde predicciones guardadas, sin reentrenar."""
import argparse
import json
from pathlib import Path

import numpy as np
from src.metricas import comparar_distribuciones, resumen_regresion


def evaluar(carpeta, salida):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    carpeta, salida = Path(carpeta), Path(salida)
    with np.load(carpeta / 'predicciones_test.npz') as d:
        y, pred = d['y_test'], d['y_hat_test']
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Se requieren vectores finitos, no vacíos y alineados')
    metadata = json.loads((carpeta / 'metadata.json').read_text(encoding='utf-8'))
    salida.mkdir(parents=True, exist_ok=True)
    metricas = resumen_regresion(y, pred)
    tabla = comparar_distribuciones(y, pred)
    tabla.to_csv(salida / 'distribucion_lgd.csv', index_label='estadistico')
    (salida / 'metricas_lgd.json').write_text(json.dumps(metricas, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    limites = (min(0, y.min(), pred.min()), max(1, y.max(), pred.max()))
    bins = np.linspace(*limites, 31)
    for valores, nombre in [(y, 'Observada'), (pred, 'Predicha')]:
        axes[0].hist(valores, bins=bins, density=True, alpha=.5, label=nombre)
        orden = np.sort(valores)
        axes[1].step(orden, np.arange(1, len(orden)+1)/len(orden), where='post', label=nombre)
    axes[0].set(title='Distribución de LGD en prueba', xlabel='LGD', ylabel='Densidad')
    axes[1].set(title='Distribución acumulada en prueba', xlabel='LGD', ylabel='Proporción acumulada')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(salida / 'figura_1_distribucion_lgd.png', dpi=180)
    plt.close(fig)
    modo = 'DEMOSTRACIÓN: muestra reducida' if metadata.get('demo') else 'Corrida sin --demo'
    texto = f'''# Evaluación de LGD

**{modo}**. Fuente: predicciones de prueba guardadas en `{carpeta}`.

Se evaluaron {len(y)} créditos. MAE = {metricas['mae']:.6f}; RMSE = {metricas['rmse']:.6f}.
El MAE equivale a {100*metricas['mae']:.2f} puntos porcentuales de LGD.

La Figura 1 compara el histograma y las distribuciones acumuladas en las mismas filas de prueba. La Tabla 1 (`distribucion_lgd.csv`) contiene media, desviación estándar y cuantiles. Fuente de ambas: cálculo propio sobre esas predicciones.

La media observada es {y.mean():.6f} y la predicha {pred.mean():.6f}; las desviaciones estándar son {y.std():.6f} y {pred.std():.6f}. Una dispersión predicha menor indica concentración alrededor de la media; no demuestra por sí sola un error de implementación, pues MSE estima una media condicional, no una simulación de la distribución individual de pérdidas.

Estos resultados no sustituyen la comparación con la referencia bajo iguales datos y particiones ni el resumen de 30 semillas. No use prueba para elegir arquitectura.
'''
    (salida / 'evaluacion_lgd.md').write_text(texto, encoding='utf-8')
    return metricas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo', default='models/lgd')
    p.add_argument('--salida', default='reports/lgd')
    args = p.parse_args()
    print(json.dumps(evaluar(args.modelo, args.salida), indent=2))


if __name__ == '__main__':
    main()
