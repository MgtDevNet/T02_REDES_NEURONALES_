# Punto 2 — Pérdida dado el Incumplimiento (LGD) y Pérdida Esperada ($E[L]$)

Esta guía explica de forma clara y sencilla cómo ejecutar las funciones del **Punto 2** y qué contenido se encuentra en cada uno de los notebooks del proyecto.

---

## 1. ¿Qué se hace en el Punto 2?

1. **Modelar la Severidad de Pérdida ($\mathrm{LGD}$):** Se estima el porcentaje del crédito que no se recupera una vez que el deudor entra en incumplimiento ($\mathrm{LGD} \in [0, 1]$), comparando un modelo de referencia lineal (OLS) con un Perceptrón Multicapa (MLP) programado desde cero en NumPy.
2. **Aplicar la Regla de Comparación Justa:** Mismas particiones fijas (60% entrenamiento, 20% validación, 20% prueba), presupuesto explícito de búsqueda en validación y evaluación de robustez a través de **30 semillas aleatorias independientes**.
3. **Calcular la Pérdida Esperada ($E[L]$):** Se combinan las redes neuronales de probabilidad de impago ($\mathrm{PD}$) y severidad ($\mathrm{LGD}$) junto con la exposición ($\mathrm{EAD}$) para estimar el riesgo monetario crédito por crédito y a nivel portafolio:
   $$\widehat{E[L]}_i = \widehat{\mathrm{PD}}_i \cdot \widehat{\mathrm{LGD}}_i \cdot \mathrm{EAD}_i$$

---

## 2. ¿Qué se espera encontrar en los notebooks?

Los notebooks están organizados por etapa temática en las carpetas `Analisis_descriptivo/` y `Modelos/`:

| Notebook | Ubicación | Contenido principal y resultados que genera |
| :--- | :--- | :--- |
| **`analisis_descriptivo_LGD.ipynb`** | `Analisis_descriptivo/` | • Filtrado de créditos en default (`Charged Off` y `Default`, 46.467 registros).<br>• Justificación de por qué no se usa `out_prncp` y cálculo de LGD observada.<br>• Pruebas ANOVA y Tukey HSD para agrupar las calificaciones de riesgo `F` y `G`. |
| **`modelo_regresion_lineal_LGD.ipynb`** | `Modelos/` | • Ajuste de la regresión lineal por mínimos cuadrados ordinarios (OLS) con variables indicadoras.<br>• Tabla de significancia estadística, $R^2$, MAE ($0{,}1425$) y RMSE ($0{,}1862$).<br>• Detección de predicciones fuera del soporte $[0, 1]$. |
| **`Punto_2_30_semillas_lgd.ipynb`** | `Modelos/` | • Presupuesto de búsqueda de hiperparámetros sobre la partición de validación (semilla 42).<br>• Reentrenamiento de la red ganadora a lo largo de **30 semillas independientes** (42 a 71).<br>• Comparación de distribuciones (cuantiles y CDFs) y justificación teórica de la compresión de varianza.<br>• Genera las figuras `figura_variabilidad_30_semillas.png` y `figura_comparacion_distribuciones_lgd.png`. |
| **`Punto_2_perdida_esperada.ipynb`** | `Modelos/` | • Carga simultánea de los modelos entrenados de PD y LGD.<br>• Cálculo de la pérdida esperada sobre una cohorte representativa de 10.000 préstamos.<br>• Balance de cartera (Tasa de pérdida agregada del $10{,}02\,\%$), curva de concentración de Lorenz y dispersión $PD$ vs. $LGD$.<br>• Genera `figura_perdida_esperada_analisis.png` y `figura_dispersion_pd_lgd_el.png`. |
| **`Punto_2_completo.ipynb`** | `Modelos/` | • Notebook integral que consolida todo el flujo de trabajo del Punto 2 de principio a fin en un único archivo ejecutable. |

---

## 3. ¿Cómo ejecutar las funciones y scripts correctamente?

Tiene dos formas equivalentes de ejecutar el código:

### Opción A: Ejecución interactiva desde los Notebooks (Recomendada para explorar)

1. Abra el notebook deseado en **VS Code**, **Jupyter Lab** o **Jupyter Notebook**.
2. Asegúrese de tener seleccionado el kernel del entorno virtual (`.venv` o Python 3.11/3.12).
3. Seleccione la opción **"Run All"** (Ejecutar todas las celdas). Los notebooks ya contienen las rutas relativas configuradas para leer los datos procesados en `data/processed/` o generarlos si no existen.

---

### Opción B: Ejecución modular por línea de comandos (Scripts)

Desde la raíz del repositorio (`T02_REDES_NEURONALES_`), puede ejecutar cualquiera de las etapas de forma automatizada mediante los siguientes comandos:

#### 1. Preprocesar y partir los datos para LGD:
Filtra la población de default, calcula la LGD observada y guarda las particiones fijas en `data/processed/lgd_limpio.parquet`:
```bash
python scripts/parte2_lgd/preparar_lgd.py
```

#### 2. Entrenar la red neuronal y buscar hiperparámetros:
Ejecuta la búsqueda sobre el conjunto de validación, entrena la red ganadora y guarda el modelo en `models/lgd/red.npz`:
```bash
python scripts/parte2_lgd/entrenar_lgd.py
```

#### 3. Evaluar la regla de comparación justa (OLS vs. Red en 30 semillas):
Corre las 30 semillas independientes, genera las tablas estadísticas y exporta los gráficos de distribuciones a `reports/`:
```bash
python scripts/parte2_lgd/evaluar_lgd.py
```

#### 4. Estimar la pérdida esperada de cartera ($E[L]$):
Combina las predicciones de PD y LGD sobre la cohorte de prueba y guarda el consolidado en `reports/perdida_esperada.csv`:
```bash
python scripts/parte2_lgd/estimar_perdida.py
```

---

### Opción C: Ejecución directa en código Python

Si desea invocar las funciones directamente dentro de su propio script o terminal interactivo:

```python
import pandas as pd
from src.experimentos_lgd import ejecutar_busqueda_y_semillas_lgd
from src.perdida_esperada import estimar_perdida_esperada_cartera

# 1. Cargar datos preprocesados
datos = pd.read_parquet("data/processed/lgd_limpio.parquet")

# 2. Ejecutar la búsqueda y las 30 semillas
resultados = ejecutar_busqueda_y_semillas_lgd(
    datos,
    semilla_busqueda=42,
    semillas_evaluacion=list(range(42, 72))
)

# 3. Estimar la pérdida esperada de cartera
resumen_cartera, df_perdidas = estimar_perdida_esperada_cartera(
    ruta_modelo_pd="models/pd/red.npz",
    ruta_modelo_lgd="models/lgd/red.npz",
    tamano_muestra=10000
)

print(f"Tasa de pérdida esperada de cartera: {resumen_cartera['tasa_perdida_esperada']*100:.2f}%")
```

---

## 4. Archivos generados y reportes

Al ejecutar los scripts o notebooks, los resultados clave quedan almacenados automáticamente en:
- `models/lgd/`: Modelo neuronal entrenado (`red.npz`), preprocesador (`preprocesador.joblib`) y metadata (`metadata.json`).
- `reports/lgd/` y `reports/experimentos_lgd/`: Tablas de resultados en formato CSV (`metricas_ols.csv`, `metricas_semillas.csv`, `comparacion_cuantiles.csv`).
- `reports/`: Figuras de alta resolución para el informe final (`figura_variabilidad_30_semillas.png`, `figura_comparacion_distribuciones_lgd.png`, `figura_perdida_esperada_analisis.png`, `figura_dispersion_pd_lgd_el.png`).
