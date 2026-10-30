# Punto 1 — Probabilidad de incumplimiento (PD): cómo replicarlo desde cero

Esta guía lleva a cualquier persona desde un computador sin nada instalado hasta los mismos
resultados del informe. No hace falta abrir ni modificar el código: todo se ejecuta con
comandos de una línea y los hiperparámetros se cambian en `config.ini`.

**Qué hace el Punto 1:** entrena una red neuronal (perceptrón multicapa escrito desde cero en
NumPy) que estima la probabilidad de que un cliente no pague su crédito, la compara con una
regresión logística, la evalúa en un conjunto de prueba y la convierte en una scorecard.

---

## 1. Lo que necesita

- **Git**, para descargar el repositorio.
- **Python 3.12.** Con otra versión las librerías pueden no instalarse.
- **No necesita descargar los datos de Kaggle.** Los datos ya procesados
  (`data/processed/pd_limpio.parquet`) y los modelos entrenados (`models/`) vienen en el
  repositorio.

---

## 2. Descargar el repositorio e instalar

**Linux, macOS o WSL**

```bash
git clone https://github.com/MgtDevNet/T02_REDES_NEURONALES_.git
cd T02_REDES_NEURONALES_
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell o CMD)**

```bat
git clone https://github.com/MgtDevNet/T02_REDES_NEURONALES_.git
cd T02_REDES_NEURONALES_
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si PowerShell no deja activar el entorno, ejecute una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Para comprobar que todo quedó bien:**

```bash
python -m pytest -q
```

Las 68 pruebas deben pasar (toma unos segundos).

> **Dos reglas para todos los comandos de esta guía**
> 1. Ejecútelos desde la **raíz del repositorio** (la carpeta `T02_REDES_NEURONALES_`), no desde `Punto_1/`.
> 2. Active el entorno en cada terminal nueva (`source .venv/bin/activate` o `.venv\Scripts\activate`).

---

## 3. El corazón: entrenar y evaluar la red con un comando

```bash
python -m scripts.parte1_pd.entrenar_pd
```

Este comando hace todo el ciclo de la red:

1. Lee los hiperparámetros de `config.ini`.
2. Prepara los datos (imputación, escalado y one-hot, aprendidos solo con entrenamiento).
3. Entrena la red y se detiene cuando deja de mejorar en validación (parada temprana).
4. Mide AUC, Gini, KS y Brier en entrenamiento, validación y prueba.
5. Guarda el modelo y su scorecard en `models/pd` (es el que usa la app).

Tarda unos segundos. Al final imprime una tabla como esta, que debe coincidir:

```
           n  tasa_malos    auc   gini     ks  brier
train  31505      0.1229 0.6926 0.3853 0.2784 0.1016
valid  14967      0.1378 0.6383 0.2766 0.2009 0.1156
test   28458      0.1361 0.6251 0.2502 0.1854 0.1151
```

La fila que importa es **`test`**: créditos de julio a diciembre de 2012 que el modelo nunca vio.

---

## 4. Replicar todos los resultados del informe

Ejecute los pasos en este orden. Cada uno deja sus resultados en `reports/` o `models/`.

| Paso | Comando | Qué produce | Tiempo aprox. |
|---|---|---|---|
| 1 | `python -m scripts.parte1_pd.buscar_hiperparametros_pd` | Búsqueda aleatoria de 40 redes y 5 logísticas: `reports/busqueda_pd.csv` | 10 min |
| 2 | `python -m scripts.parte1_pd.entrenar_pd` | La red final: `models/pd` | segundos |
| 3 | `python -m scripts.parte1_pd.entrenar_pd --set red.capas_ocultas= --set red.dropout=0 --set red.l2=0.001 --set salida.carpeta=models/pd_logistica` | La regresión logística en NumPy: `models/pd_logistica` | segundos |
| 4 | `python -m scripts.parte1_pd.repetir_semillas_pd` | 30 corridas de la red: `reports/semillas_pd_mlp.csv` | 3 min |
| 5 | `python -m scripts.parte1_pd.repetir_semillas_pd --nombre logistica --set red.capas_ocultas= --set red.dropout=0 --set red.l2=0.001` | 30 corridas de la logística: `reports/semillas_pd_logistica.csv` | 1 min |
| 6 | `python -m scripts.parte1_pd.evaluar_pd` | Evaluación en prueba, bootstrap y figuras ROC, KS y calibración | 1–2 min |
| 7 | `python -m scripts.parte1_pd.interpretar_pd` | Importancia de variables y dependencia parcial de la red | 1–2 min |
| 8 | `python -m scripts.parte1_pd.scorecard_pd` | Tabla de puntos por variable: `reports/scorecard_pd.csv` | segundos |
| 9 | `python -m scripts.parte1_pd.figuras_scorecard_pd` | Figuras de la scorecard | segundos |
| 10 | `python -m scripts.parte1_pd.figuras_variables_riesgo_pd` | Variables de riesgo: logística frente a la red | 1–2 min |
| 11 | `python -m scripts.parte1_pd.figura_particion_pd` | Línea de tiempo de la partición | segundos |

El paso 1 es opcional: solo justifica la configuración que ya está en `config.ini`.

**Resultados que debe obtener** (en prueba):

| Modelo | Parámetros | AUC | KS | Brier |
|---|---|---|---|---|
| Red MLP 64-32 | 4.097 | 0,6251 | 0,1854 | 0,1151 |
| Regresión logística (NumPy) | 31 | 0,6296 | 0,1894 | 0,1150 |

| 30 semillas, AUC en prueba | Media | Desviación | Peor | Mejor |
|---|---|---|---|---|
| Red | 0,625 | 0,0031 | 0,618 | 0,633 |
| Logística | 0,629 | 0,0007 | 0,627 | 0,630 |

Diferencia de AUC red − logística (bootstrap pareado): −0,003, IC 95 % [−0,006; −0,0004].

Los resultados son reproducibles porque la semilla está fija (`semilla = 42`). En otro
computador pueden cambiar en el cuarto decimal por diferencias de redondeo; los tiempos
dependen de la máquina.

---

## 5. Cambiar hiperparámetros

Hay dos formas, y ninguna toca el código:

- **Permanente:** editar `config.ini`, guardar y volver a ejecutar.
- **Solo una vez:** agregar `--set seccion.clave=valor` al comando. La sección se escribe sin el
  prefijo `pd.` (por ejemplo, `[pd.red] dropout` se cambia con `--set red.dropout=0.5`).

| Sección de `config.ini` | Qué controla | Valores actuales |
|---|---|---|
| `[general]` | Semilla | `semilla = 42` |
| `[pd.red]` | Arquitectura | `capas_ocultas = 64, 32`, `activacion = relu`, `l2 = 0.0014`, `dropout = 0.3` |
| `[pd.entrenamiento]` | Cómo aprende | `optimizador = adam`, `tasa = 0.0026`, `epocas = 200`, `tam_lote = 512`, `paciencia = 10` |
| `[pd.scorecard]` | Escala de puntos | `puntaje_base = 600`, `odds_base = 50`, `pdo = 20` |
| `[pd.salida]` | Dónde se guarda el modelo | `carpeta = models/pd` |

Ejemplos:

```bash
# otra arquitectura
python -m scripts.parte1_pd.entrenar_pd --set red.capas_ocultas=64,32,16 --set red.activacion=tanh

# más regularización
python -m scripts.parte1_pd.entrenar_pd --set red.dropout=0.5 --set red.l2=0.01

# otro optimizador y otra tasa de aprendizaje
python -m scripts.parte1_pd.entrenar_pd --set entrenamiento.optimizador=sgd --set entrenamiento.tasa=0.01

# otra semilla
python -m scripts.parte1_pd.entrenar_pd --set semilla=7
```

**Para probar sin dañar los modelos de la app**, agregue `--set salida.carpeta=models/prueba`.

**Versión rápida** (20 % de los datos, unos segundos): agregue `--demo`.

**Evaluación completa de una red nueva** (curvas, calibración, bootstrap y 30 semillas):

```bash
python -m scripts.parte1_pd.entrenar_pd --set red.capas_ocultas=32,16 --set salida.carpeta=models/prueba
python -m scripts.parte1_pd.evaluar_pd --modelos models/pd_logistica models/prueba
python -m scripts.parte1_pd.repetir_semillas_pd --nombre prueba --set red.capas_ocultas=32,16 --semillas 5
```

`evaluar_pd` sobrescribe los archivos de evaluación en `reports/`. Para volver a los originales:
`git checkout reports/ models/`.

### Ejemplos para explorar y qué esperar

Resultados reales con todos los datos y la semilla 42. Cada fila es un cambio frente a la
configuración actual, que da AUC **0,693** en entrenamiento y **0,625** en prueba (mejor época 10).
Todos se ejecutan así, cambiando solo los `--set`:

```bash
python -m scripts.parte1_pd.entrenar_pd --set red.dropout=0 --set salida.carpeta=models/prueba
```

**Regularización: el equilibrio entre memorizar y generalizar**

| Cambio (`--set`) | Qué esperar | AUC entren. → prueba |
|---|---|---|
| `red.dropout=0` | Memoriza más: sube en entrenamiento, baja en prueba | 0,696 → 0,622 |
| `red.dropout=0.6` | Le cuesta aprender: baja en ambos | 0,680 → 0,622 |
| `red.l2=0` | Sin penalización: más sobreajuste | 0,699 → 0,622 |
| `red.l2=0.1` | Pesos casi en cero: la red no aprende | 0,635 → 0,591 |

Poca regularización sobreajusta y demasiada no deja aprender. La configuración actual
(dropout 0,3 y L2 0,0014) está en el punto medio.

**Tamaño de la red: más grande no es mejor**

| Cambio (`--set`) | Parámetros | AUC entren. → prueba | Tiempo |
|---|---|---|---|
| `red.capas_ocultas=8` | 257 | 0,686 → 0,628 | 0,9 s |
| (actual: 64, 32) | 4.097 | 0,693 → 0,625 | 2,6 s |
| `red.capas_ocultas=256,128` | 40.961 | 0,694 → 0,625 | 11,5 s |
| `red.capas_ocultas=256,128` `red.dropout=0` `red.l2=0` | 40.961 | 0,726 → 0,615 | 5,5 s |
| `red.capas_ocultas=` `red.dropout=0` `red.l2=0.001` (logística) | 31 | 0,681 → 0,630 | 1,2 s |

Una red pequeña, o sin capas ocultas, rinde igual o mejor que una de 40 mil parámetros. La red
grande sin regularizar es el ejemplo más claro de sobreajuste: el mejor AUC en entrenamiento y el
peor en prueba.

**Activación**

| Cambio (`--set`) | Qué esperar | AUC en prueba |
|---|---|---|
| `red.activacion=tanh` | Parecido a ReLU, aprende más despacio | 0,627 (mejor época 19) |
| `red.activacion=sigmoide` | Gradientes pequeños: aprende peor | 0,615 |

**Optimizador y tasa de aprendizaje**

| Cambio (`--set`) | Qué esperar | AUC en prueba | Mejor época |
|---|---|---|---|
| `entrenamiento.tasa=0.05` | Pasos demasiado grandes: se pasa del mínimo | 0,616 | 3 |
| `entrenamiento.tasa=0.00005` | Pasos diminutos: usa las 200 épocas y no termina | 0,622 | 199 de 200 |
| `entrenamiento.optimizador=sgd` | SGD con la tasa de Adam es muy lento | 0,588 | 200 de 200 |
| `entrenamiento.optimizador=sgd` `entrenamiento.tasa=0.1` | Con una tasa adecuada, SGD se acerca a Adam | 0,624 | 77 |

La tasa de aprendizaje es el hiperparámetro más delicado. Adam funciona con tasas pequeñas porque
ajusta el paso de cada peso; SGD necesita una tasa mucho mayor.

**Entrenamiento y semilla**

| Cambio (`--set`) | Qué esperar | AUC en prueba |
|---|---|---|
| `entrenamiento.tam_lote=32` | Más actualizaciones por época, más ruido | 0,624 (mejor época 3) |
| `entrenamiento.paciencia=1` | Se detiene al primer tropiezo, antes de tiempo | 0,619 (5 épocas) |
| `semilla=7` | Misma configuración, otra inicialización | 0,625 (mejor época 17) |

Con otra semilla el AUC queda casi igual, pero la mejor época cambia de 10 a 17. Por eso se
reportan 30 semillas y no una sola corrida.

**Tres ejemplos para mostrar en vivo**

1. `--set red.capas_ocultas=256,128 --set red.dropout=0 --set red.l2=0`: sobreajuste a la vista (0,726 en entrenamiento, 0,615 en prueba).
2. `--set red.l2=0.1`: demasiada regularización, la red no aprende (0,591).
3. `--set red.capas_ocultas= --set red.dropout=0 --set red.l2=0.001`: la regresión logística gana con 31 parámetros (0,630).

Los tiempos dependen del computador. Con `--demo` todo corre más rápido, pero los números cambian
porque se usa solo el 20 % de los datos.

---

## 6. Opcional: rehacer los datos desde Kaggle

Solo si quiere repetir también la limpieza:

1. Descargue *Credit Risk Analysis* (Lending Club) de
   https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset y descomprímalo en `datos/`
   (debe quedar `datos/loan/loan.csv`).
2. Ejecute `python -m scripts.parte1_pd.preparar_datos_pd`. Genera `data/processed/pd_limpio.parquet`:
   solo créditos a 36 meses, con partición temporal (entrenamiento jun-2007 a dic-2011,
   validación ene–jun 2012, prueba jul–dic 2012).
3. Vuelva a ejecutar el notebook `Punto_1/Primer_Punto.ipynb` (la regresión logística de
   referencia) para que sus predicciones queden alineadas con los nuevos datos. El notebook necesita
   Jupyter y algunas librerías adicionales: `pip install jupyter seaborn statsmodels`.

---

## 7. La aplicación web

```bash
streamlit run app/app.py
```

Se abre en http://localhost:8501. Usa los modelos guardados en `models/` y no reentrena.

---

## Problemas comunes

| Síntoma | Solución |
|---|---|
| `No module named 'src'` | Ejecute desde la raíz del repositorio y con `python -m scripts.parte1_pd.…` |
| `No module named 'numpy'` (u otra librería) | Active el entorno: `source .venv/bin/activate` |
| `evaluar_pd` dice que las predicciones del notebook no están en el mismo orden | Cambió `pd_limpio.parquet`: vuelva a ejecutar `Punto_1/Primer_Punto.ipynb` |
| La app muestra otros números | Se reentrenó sobre `models/`: `git checkout models/` |
