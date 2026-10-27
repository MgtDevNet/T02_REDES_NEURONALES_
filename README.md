# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

Redes neuronales y algoritmos bioinspirados — Universidad Nacional de Colombia, 2026-02.

## Instalación

Requiere Python 3.12. Desde la carpeta del repositorio:

**Linux / macOS / WSL**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell o CMD)**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si PowerShell no permite activar el entorno, ejecutar una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. El entorno se activa en cada
terminal nueva antes de ejecutar los comandos de abajo; en Windows, donde aquí dice
`python3.12` o `source .venv/bin/activate` se usan los comandos de este bloque.

## Datos

Conjunto *Credit Risk Analysis* (Lending Club) de Kaggle:
https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

Descargar el zip y descomprimirlo en `datos/` (debe quedar `datos/loan/loan.csv`
y `datos/LCDataDictionary.xlsx`). Los datos procesados que usan los modelos se
guardan en `data/processed/`.

## Cómo ejecutar

Todos los hiperparámetros están en **`config.ini`**. Se edita un valor, se guarda y se vuelve
a ejecutar; no hay que tocar el código.

### Parte 1, de punta a punta

```bash
python preparar_datos_pd.py          # loan.csv -> data/processed/pd_limpio.parquet (ya incluido en el repo)
python entrenar_pd.py                # MLP final -> models/pd
python entrenar_pd.py --set red.capas_ocultas= --set red.dropout=0 \
                      --set salida.carpeta=models/pd_logistica      # logística en NumPy (la usa la app)
python evaluar_pd.py                 # prueba: MLP frente a la logística de Primer_Punto.ipynb (sin reentrenar)
python interpretar_pd.py             # importancia por permutación y dependencia parcial
python scorecard_pd.py               # puntos por variable (logística); --modelo models/pd para la red
python buscar_hiperparametros_pd.py  # búsqueda aleatoria (~10 min) -> reports/busqueda_pd.csv
python repetir_semillas_pd.py        # 30 semillas de la configuración final -> reports/
```

Datos de la Parte 1: solo créditos a 36 meses, con partición temporal por fecha de
originación (train jun-2007 a dic-2011, valid ene-jun 2012, test jul-dic 2012). La
justificación está en el docstring de `preparar_datos_pd.py` y en el reporte.

### Comandos generales

```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python entrenar_pd.py            # Parte 1: probabilidad de incumplimiento + scorecard
python entrenar_pd.py --demo     # versión rápida con una muestra (< 2 minutos)
python entrenar_lgd.py           # Parte 2: pérdida dado incumplimiento
python entrenar_lgd.py --demo
```

Para cambiar un valor solo en una ejecución, sin editar `config.ini`, se usa `--set`
con la forma `seccion.clave=valor` (la sección sin el prefijo `pd.` / `lgd.`):

```bash
python entrenar_pd.py --set red.capas_ocultas=64,32 --set red.dropout=0.3
python entrenar_pd.py --set entrenamiento.tasa=0.01 --set semilla=7
python entrenar_pd.py --set scorecard.pdo=40
python entrenar_pd.py --set red.capas_ocultas= --set red.dropout=0 \
                      --set salida.carpeta=models/pd_logistica      # regresión logística
python entrenar_lgd.py --set red.perdida=entropia_cruzada           # respuesta fraccional
python entrenar_lgd.py --set red.capas_ocultas= --set red.salida=identidad \
                       --set red.dropout=0 --set salida.carpeta=models/lgd_lineal   # regresión lineal
```

La red neuronal está implementada desde cero con NumPy en `src/red_neuronal.py`.

## Aplicación web

```bash
streamlit run app/app.py         # se abre en http://localhost:8501
```

Una persona ingresa sus datos y obtiene su puntaje, su probabilidad de incumplimiento, su
percentil frente a la población, qué características le suman o restan puntos y cómo cambiaría su
puntaje si cambiara alguna. La página «Cómo funciona» explica el modelo. Carga los modelos de
`models/` (no reentrena). El modelo principal y los enlaces al reporte y al video se configuran en
la sección `[app]` de `config.ini`.

## Pruebas

```bash
python -m pytest -q
```

## Estructura

```
config.ini         hiperparámetros de todos los experimentos
entrenar_pd.py     entrenamiento de la Parte 1 (un solo comando)
entrenar_lgd.py    entrenamiento de la Parte 2 (un solo comando)
datos/             datos originales de Kaggle (no se suben al repo)
data/processed/    datos procesados que usan los modelos
src/               código: limpieza, red neuronal, scorecard, métricas
models/            modelos entrenados (la app los carga sin reentrenar)
app/               aplicación web (Streamlit): app.py, evaluar.py, como_funciona.py, motor.py
reports/           resultados de la Parte 1 (búsqueda, semillas, evaluación, scorecard)
reports/figures/   figuras regenerables con evaluar_pd.py e interpretar_pd.py
```

## Red neuronal del punto 2

Para preparar la LGD desde `loan.csv` y entrenar con la red de Manuel, consulte [Punto_2/RED_NEURONAL.md](Punto_2/RED_NEURONAL.md).

