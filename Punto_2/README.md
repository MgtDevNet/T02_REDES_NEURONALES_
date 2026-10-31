# Punto 2 — Pérdida dado el Incumplimiento (LGD) y Pérdida Esperada ($E[L]$)

Esta guía explica de forma clara y precisa cómo ejecutar las funciones del **Punto 2** y qué contenido se encuentra en cada uno de los notebooks y scripts del proyecto.

---

## 1. ¿Qué se hace en el Punto 2?

1. **Modelar la Severidad de Pérdida ($\mathrm{LGD}$):** Se estima el porcentaje del crédito que no se recupera una vez que el deudor entra en incumplimiento ($\mathrm{LGD} \in [0, 1]$), comparando un modelo de referencia lineal (OLS) con un Perceptrón Multicapa (MLP) programado desde cero en NumPy con salida sigmoide.
2. **Aplicar la Regla de Comparación Justa:** Mismas particiones fijas (60% entrenamiento, 20% validación, 20% prueba), presupuesto explícito de búsqueda en validación y evaluación de robustez a través de **30 semillas aleatorias independientes**.
3. **Calcular la Pérdida Esperada ($E[L]$):** Se combinan las redes neuronales de probabilidad de impago ($\mathrm{PD}$) y severidad ($\mathrm{LGD}$) junto con la exposición ($\mathrm{EAD}$) para estimar el riesgo monetario crédito por crédito y a nivel portafolio:
   $$\widehat{E[L]}_i = \widehat{\mathrm{PD}}_i \cdot \widehat{\mathrm{LGD}}_i \cdot \mathrm{EAD}_i$$

---

## 2. ¿Qué se espera encontrar en los notebooks?

Los notebooks están organizados por etapa temática en las carpetas `Analisis_descriptivo/` y `Modelos/`:

| Notebook | Ubicación | Contenido principal y resultados que genera |
| :--- | :--- | :--- |
| **`analisis_descriptivo_LGD.ipynb`** | `Analisis_descriptivo/` | • Filtrado de créditos en default definitivo (`Charged Off` y `Default`, 46.467 registros).<br>• Justificación de por qué no se usa `out_prncp` y cálculo de LGD observada.<br>• Pruebas ANOVA y Tukey HSD para agrupar las calificaciones de riesgo `F` y `G`. |
| **`modelo_regresion_lineal_LGD.ipynb`** | `Modelos/` | • Ajuste de la regresión lineal por mínimos cuadrados ordinarios (OLS) con variables indicadoras.<br>• Tabla de significancia estadística, $R^2$, MAE ($0{,}1425$) y RMSE ($0{,}1862$).<br>• Detección de predicciones fuera del soporte $[0, 1]$. |
| **`Punto_2_30_semillas_lgd.ipynb`** | `Modelos/` | • Presupuesto de búsqueda de hiperparámetros sobre la partición de validación (semilla 42).<br>• Reentrenamiento de la red ganadora a lo largo de **30 semillas independientes** (semillas 42 a 71 con MAE medio de $0{,}141219$, consistente con las semillas 0 a 29 de `src/experimentos_lgd.py` con MAE de $0{,}141308$).<br>• Comparación de distribuciones (cuantiles y CDFs) y justificación teórica de la compresión de varianza.<br>• Genera las figuras `figura_variabilidad_30_semillas.png` y `figura_comparacion_distribuciones_lgd.png`. |
| **`Punto_2_perdida_esperada.ipynb`** | `Modelos/` | • Carga simultánea de los modelos entrenados de PD y LGD.<br>• Estimación de la pérdida esperada sobre la cohorte de prueba de créditos a 36 meses con estado definitivo (10.000 créditos muestreados representativos).<br>• Balance de cartera (tasa de pérdida agregada del $8{,}85\,\%$ en prueba), curva de concentración de Lorenz y dispersión $PD$ vs. $LGD$.<br>• Genera `figura_perdida_esperada_analisis.png` y `figura_dispersion_pd_lgd_el.png`. |
| **`Punto_2_completo.ipynb`** | `Modelos/` | • Notebook integral que consolida todo el flujo de trabajo del Punto 2 de principio a fin en un único archivo ejecutable. |

---

## 3. ¿Cómo ejecutar las funciones y scripts correctamente?

Tiene tres formas equivalentes de ejecutar el código:

### Opción A: Ejecución interactiva desde los Notebooks (Recomendada para explorar)

1. Abra el notebook deseado en **VS Code**, **Jupyter Lab** o **Jupyter Notebook**.
2. Asegúrese de tener seleccionado el kernel del entorno virtual (`.venv` o Python 3.11/3.12 con los paquetes de `requirements.txt`).
3. Seleccione la opción **"Run All"** (Ejecutar todas las celdas). Los notebooks ya contienen las rutas relativas configuradas para leer los datos procesados en `data/processed/` o el archivo original en `datos/loan/loan.csv`, `Punto_2/loan/loan.csv` o `loan.csv`.

---

### Opción B: Ejecución modular por línea de comandos (Scripts)

Desde la raíz del repositorio (`T02_REDES_NEURONALES_`), ejecute cualquiera de las etapas de forma automatizada mediante la sintaxis de módulo (`python -m`):

#### 1. Preprocesar y partir los datos para LGD:
Filtra la población en default definitivo (`Charged Off` y `Default`), calcula la LGD observada y guarda las particiones fijas (60/20/20) en `data/processed/lgd_limpio.parquet`:
```bash
python -m scripts.parte2_lgd.preparar_lgd --csv datos/loan/loan.csv
```

#### 2. Entrenar la red neuronal de LGD:
Entrena la arquitectura neuronal elegida (configurada en `config.ini` o pasada vía `--set`) y guarda los artefactos en `models/lgd/` (`red.npz`, `preprocesador.joblib`, `predicciones_test.npz` y `metadata.json`):
```bash
python -m scripts.parte2_lgd.entrenar_lgd
```
*(Nota: Para ejecutar la búsqueda de hiperparámetros y las 30 réplicas completas, use el notebook `Punto_2_30_semillas_lgd.ipynb` o el módulo `src.experimentos_lgd`).*

#### 3. Evaluar las predicciones de prueba (sin reentrenar):
Evalúa las predicciones guardadas en `models/lgd/predicciones_test.npz`, calculando MAE, RMSE, generando la tabla de cuantiles `distribucion_lgd.csv` y la gráfica `figura_1_distribucion_lgd.png` en `reports/lgd/`:
```bash
python -m scripts.parte2_lgd.evaluar_lgd
```

#### 4. Estimar la pérdida esperada de cartera ($E[L]$):
Combina los modelos entrenados de PD y LGD sobre una cartera de créditos y genera el consolidado crédito por crédito en `reports/perdida_esperada.csv`:
```bash
python -m scripts.parte2_lgd.estimar_perdida --creditos data/processed/cohorte_perdida.parquet
```

---

### Opción C: Ejecución directa en código Python

Si desea invocar las funciones directamente dentro de su propio script o terminal interactivo:

```python
import pandas as pd
from src.experimentos_lgd import ejecutar_busqueda_y_semillas_lgd
from src.perdida_esperada import estimar_perdida_esperada_cartera

# 1. Cargar datos preprocesados de LGD
datos = pd.read_parquet("data/processed/lgd_limpio.parquet")

# 2. Ejecutar la búsqueda y las 30 semillas
resultados = ejecutar_busqueda_y_semillas_lgd(
    datos,
    semilla_busqueda=42,
    semillas_evaluacion=list(range(42, 72))
)

# 3. Estimar la pérdida esperada sobre la cohorte de prueba
resumen_cartera, df_perdidas = estimar_perdida_esperada_cartera(
    ruta_modelo_pd="models/pd/red.npz",
    ruta_modelo_lgd="models/lgd/red.npz",
    tamano_muestra=10000
)

print(f"Tasa de pérdida esperada de cartera: {resumen_cartera['tasa_perdida_esperada']*100:.2f}%")
print(f"Pérdida esperada media por crédito: ${resumen_cartera['el_media_por_credito']:,.2f} USD")
```

---

## 4. Archivos generados y reportes

Al ejecutar los scripts o notebooks, los resultados clave quedan almacenados automáticamente en:
- `models/lgd/`: Modelo neuronal entrenado (`red.npz`), preprocesador ajustado con train (`preprocesador.joblib`), predicciones de prueba (`predicciones_test.npz`) y metadatos del entrenamiento (`metadata.json`).
- `reports/lgd/` y `reports/experimentos_lgd/`: Tablas de resultados en formato CSV (`referencia_ols.csv`, `resumen_semillas.csv`, `corridas.csv`, `distribucion_lgd.csv`, `busqueda_hiperparametros_lgd.csv`, `tabla_comparativa_ols_vs_mlp.csv`).
- `reports/`: Figuras de alta resolución para el informe final (`figura_variabilidad_30_semillas.png`, `figura_comparacion_distribuciones_lgd.png`, `figura_perdida_esperada_analisis.png`, `figura_dispersion_pd_lgd_el.png`) y el archivo de estimaciones a nivel de préstamo `perdida_esperada.csv`.
