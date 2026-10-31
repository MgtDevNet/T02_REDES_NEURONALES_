# Evaluación de LGD

**Corrida sin --demo**. Fuente: predicciones de prueba guardadas en `models\lgd`.

Se evaluaron 9294 créditos. MAE = 0.141461; RMSE = 0.185119.
El MAE equivale a 14.15 puntos porcentuales de LGD.

La Figura 1 compara el histograma y las distribuciones acumuladas en las mismas filas de prueba. La Tabla 1 (`distribucion_lgd.csv`) contiene media, desviación estándar y cuantiles. Fuente de ambas: cálculo propio sobre esas predicciones.

La media observada es 0.749883 y la predicha 0.749795; las desviaciones estándar son 0.201771 y 0.078844. Una dispersión predicha menor indica concentración alrededor de la media; no demuestra por sí sola un error de implementación, pues MSE estima una media condicional, no una simulación de la distribución individual de pérdidas.

Estos resultados no sustituyen la comparación con la referencia bajo iguales datos y particiones ni el resumen de 30 semillas. No use prueba para elegir arquitectura.
