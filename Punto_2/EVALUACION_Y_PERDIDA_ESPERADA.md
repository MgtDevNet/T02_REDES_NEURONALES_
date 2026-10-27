# Evaluación de LGD y pérdida esperada

## 1. MAE, RMSE y distribución

Para n créditos de prueba, con LGD observada y_i y predicción yhat_i:

- MAE = (1/n) Σ |y_i − yhat_i|.
- RMSE = √[(1/n) Σ (y_i − yhat_i)²].

Ambos se expresan en unidades de LGD; multiplicados por 100 se interpretan en puntos porcentuales. RMSE da mayor peso a errores grandes. No son porcentajes relativos de error ni porcentajes de exactitud.

**Tabla 1. Resultados preliminares de demostración.** Fuente: salida de consola compartida por la usuaria, ejecución con `--demo`, semilla 42, red 32/16, 1.859 créditos de prueba. No se ha recibido aquí el artefacto de esa corrida ni se ha verificado contra sus archivos.

| Métrica/estadístico | Observada | Predicha |
|---|---:|---:|
| Media | 0,7503 | 0,7488 |
| Desviación estándar | 0,2030 | 0,0855 |
| Percentil 5 | 0,3178 | 0,6328 |
| Mediana | 0,8128 | 0,7288 |
| Percentil 95 | 0,9762 | 0,8850 |

En esa corrida, MAE = 0,1453 y RMSE = 0,1901: errores de 14,53 y 19,01 puntos porcentuales, respectivamente. Como muestra la Tabla 1, la media predicha difiere de la observada en −0,0015 (−0,15 puntos porcentuales), pero las predicciones presentan menor dispersión. Los cuantiles extremos están comprimidos hacia el centro. Una media cercana no demuestra que la distribución completa se reproduzca bien. La distancia KS reportada entre distribuciones es 0,2550; es una distancia descriptiva, no un p-valor.

La compresión es compatible con estimar E[LGD|X] mediante MSE: la varianza de una media condicional suele ser menor que la varianza de los resultados individuales. No prueba por sí sola sobreajuste, subajuste ni un fallo del código. Hay que comparar con la referencia, revisar residuos y cuantiles y repetir el experimento bajo condiciones equivalentes.

Estas cifras son **preliminares de demo**, no resultados finales ni promedios de 30 semillas. No deben mezclarse con las métricas históricas del notebook lineal, que usa otras entradas/particiones.

Para generar el reporte, tabla y Figura 1 desde los archivos de la corrida en su PC:

```powershell
.\.venv\Scripts\python.exe -m scripts.parte2_lgd.evaluar_lgd --modelo models/lgd_demo --salida reports/lgd_demo
```

Para la corrida completa, después de entrenar sin `--demo`:

```powershell
.\.venv\Scripts\python.exe -m scripts.parte2_lgd.evaluar_lgd --modelo models/lgd --salida reports/lgd
```

El comando genera `evaluacion_lgd.md`, `metricas_lgd.json`, `distribucion_lgd.csv` y `figura_1_distribucion_lgd.png` sin reentrenar. La figura utiliza las mismas observaciones de prueba para ambos histogramas y sus distribuciones acumuladas.

## 2. Definición y justificación de EAD

La exposición histórica al incumplimiento no está reconstruida. Para mantener coherencia con el objetivo actual:

LGD_proxy = (funded_amnt − total_rec_prncp) / funded_amnt,

se elige EAD_proxy = funded_amnt, expresado en dólares según las unidades de Lending Club. Es una base de capital original constante por crédito, conocida una vez financiado. No debe confundirse con saldo vivo al incumplimiento, ni usarse como dato conocido antes de financiar si el caso de uso no lo permite: en ese momento sería necesario definir el monto propuesto como escenario y validar ese uso.

No se usa automáticamente `out_prncp`: es un saldo observado que puede aparecer en cero tras un castigo y no reconstruye por sí solo la exposición histórica. Tampoco se multiplica esta LGD, normalizada por capital original, por un saldo amortizado distinto sin redefinir y volver a estimar el objetivo; eso cambiaría la base económica.

## 3. Combinación de ambos modelos

Con D indicador de incumplimiento, X información disponible y M el monto original fijo del crédito:

E[L_proxy | X, M] = P(D=1 | X, M) × E[LGD_proxy | D=1, X, M] × M.

No se necesita asumir independencia entre incumplimiento y severidad: la segunda esperanza ya es condicional al incumplimiento. Si la EAD fuera aleatoria, la expresión general sería PD × E[LGD × EAD | D=1, X]; separar las dos esperanzas requeriría condiciones adicionales.

Ejemplo exclusivamente ilustrativo: PD = 0,10; LGD_proxy = 0,80; monto = 10.000 USD. La pérdida esperada proxy es 800 USD. No es un pronóstico observado de un crédito real ni una pérdida segura.

**Limitación de población:** PD debe predecir la clase 1 completa del enunciado. El modelo actual de LGD se entrenó solo con Charged Off y Default. Su severidad no está validada para Late (31-120 days) ni Does not meet the credit policy. Status:Charged Off. Multiplicarlos produce una aproximación con extrapolación de severidad a estados no cubiertos; para considerarlo resultado final hay que justificar esa extrapolación o ampliar la población de LGD y reentrenar. También deben ser coherentes los horizontes de PD y de observación de severidad. La proxy no incorpora explícitamente recuperaciones netas, costos completos ni descuento temporal.

## 4. Calcular por crédito sin reentrenar

`scripts/parte2_lgd/estimar_perdida.py` carga **los dos modelos de Manuel** junto con sus respectivos preprocesadores y calcula PD, LGD y pérdida para cada fila del mismo archivo. No mezcla filas de archivos de predicciones distintos ni usa la LGD observada como si fuera una predicción.

Necesita `models/pd/red.npz`, `models/pd/preprocesador.joblib` y los equivalentes de `models/lgd/`. El archivo de entrada debe contener `id` único, `funded_amnt` y las predictoras tipadas como al entrenar (plazo/empleo numéricos y antigüedad calculada). **No sirve directamente el loan.csv crudo ni el parquet de LGD actual**, que omite identificador y exposición y contiene solo la submuestra de incumplidos. No hace falta conocer la LGD observada de los clientes nuevos. Cargue únicamente artefactos joblib propios o de procedencia conocida.

```powershell
.\.venv\Scripts\python.exe -m scripts.parte2_lgd.estimar_perdida --creditos datos/creditos_para_predecir.parquet --modelo-pd models/pd --modelo-lgd models/lgd
```

Los datos de entrada y el modelo de PD todavía deben estar disponibles para ejecutar este paso con créditos reales. El programa produce `reports/perdida_esperada.csv` con una fila por crédito: `id`, `pd`, `lgd_proxy`, `ead_proxy`, `perdida_esperada_proxy`. La suma es la pérdida esperada proxy de la cartera, no una medida de pérdida inesperada ni capital económico. Para evaluar desempeño fuera de muestra, use créditos no utilizados para ajustar o seleccionar ninguno de los dos modelos.
