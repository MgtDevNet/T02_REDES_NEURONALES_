"""Página «Cómo funciona»: explica de forma intuitiva los datos, el puntaje y los modelos.
Las cifras se leen de reports/ (generados por los scripts de la Parte 1), no se escriben a mano."""

import pandas as pd
import streamlit as st

from estilo import aplicar_estilo, barra_lateral, cargar_modelo, encabezado
from formato import numero, porcentaje
from motor import RAIZ

aplicar_estilo()
modelo, config = barra_lateral()
encabezado("Cómo funciona", "Qué hay detrás de tu puntaje, explicado paso a paso.")
s = modelo.scorecard.parametros()

st.markdown("""
### 1. ¿Qué responde esta herramienta?
Antes de prestar, un banco quiere saber **qué tan probable es que no le paguen**. A eso se le llama
**probabilidad de incumplimiento (PD)**. La herramienta la estima con la información que el banco
conoce *en el momento de la solicitud*: tus ingresos, tus deudas, tu historial crediticio y para qué
es el crédito. Nada de lo que pase después del desembolso se usa, porque en la vida real no se conoce.
""")

st.markdown("""
### 2. ¿Con qué datos aprendió?
Con créditos de consumo reales de **Lending Club**, una plataforma de préstamos de Estados Unidos.
Un crédito se considera **malo** si fue castigado, quedó en impago o tiene más de 30 días de atraso.
""")
d = pd.read_parquet(RAIZ / "data/processed/pd_limpio.parquet", columns=["particion", "y"])
tabla = (d.groupby("particion")["y"].agg(["size", "mean"]).reindex(["train", "valid", "test"])
         .rename(index={"train": "Aprendizaje (jun 2007 – dic 2011)", "valid": "Ajuste (ene – jun 2012)",
                        "test": "Prueba final (jul – dic 2012)"}))
tasa_train, tasa_test = tabla["mean"].iloc[0], tabla["mean"].iloc[2]
tabla.columns = ["Créditos", "Tasa de malos"]
tabla["Tasa de malos"] = tabla["Tasa de malos"].map(porcentaje)
tabla["Créditos"] = tabla["Créditos"].map(numero)
st.table(tabla)
st.markdown("""
- **Solo créditos a 36 meses.** Los datos llegan hasta diciembre de 2015 y muchos créditos a 60 meses
  aún no habían terminado: solo se veían los que fallaron temprano, lo que exageraba su riesgo.
- **Separación por fechas.** El modelo aprende con el pasado y se califica con créditos posteriores,
  como pasaría en un banco real. La prueba final se usa una sola vez.
""")

st.markdown("### 3. De la probabilidad al puntaje")
st.markdown(f"""
El puntaje traduce la probabilidad a una escala fácil de leer: **más puntos = menos riesgo**. Se
define con tres números: **{s['puntaje_base']} puntos** equivalen a {s['odds_base']} buenos pagadores
por cada malo, y cada **{s['pdo']} puntos** esa proporción se duplica.
""")
st.latex(r"\text{odds} = \frac{1 - PD}{PD} \qquad "
         r"\text{puntaje} = \text{offset} + \text{factor} \cdot \ln(\text{odds})")
st.latex(rf"\text{{factor}} = \frac{{PDO}}{{\ln 2}} = {numero(s['factor'], 2).replace(',', '{,}')} \qquad "
         rf"\text{{offset}} = {s['puntaje_base']} - \text{{factor}} \cdot \ln({s['odds_base']}) = {numero(s['offset'], 2).replace(',', '{,}')}")
st.markdown("""
Con la regresión logística el riesgo es una suma de efectos, así que el puntaje se reparte
**exactamente** en puntos por característica: cada una te suma o te resta frente a un solicitante
promedio. Eso es lo que muestra el gráfico «¿Qué explica tu puntaje?».
""")

st.markdown("### 4. ¿Por qué regresión logística y no la red neuronal?")
# La logística es la de referencia del informe (Punto_1/Primer_Punto.ipynb), evaluada por
# scripts/parte1_pd/evaluar_pd.py; la red, con sus 30 semillas de repetir_semillas_pd.py.
filas, auc_replica = [], None
ruta = RAIZ / "reports/evaluacion_pd.csv"
if ruta.exists():
    evaluacion = pd.read_csv(ruta).query("grupo == 'total'").set_index("modelo")
    if "Regresión logística (referencia)" in evaluacion.index:
        ref = evaluacion.loc["Regresión logística (referencia)"]
        filas.append({"Modelo": "Regresión logística",
                      "AUC en prueba": f"{numero(ref['auc'], 3)} (no depende de la semilla)",
                      "Brier": numero(ref["brier"], 4),
                      "Parámetros": numero(int(ref["parametros"]))})
    if "Regresión logística (NumPy)" in evaluacion.index:
        auc_replica = evaluacion.loc["Regresión logística (NumPy)", "auc"]
ruta = RAIZ / "reports/semillas_pd_mlp.csv"
if ruta.exists():
    r = pd.read_csv(ruta)
    filas.append({"Modelo": "Red neuronal",
                  "AUC en prueba": f"{numero(r['auc_test'].mean(), 3)} ± {numero(r['auc_test'].std(), 3)} "
                                   f"({len(r)} semillas)",
                  "Brier": numero(r["brier_test"].mean(), 4),
                  "Parámetros": numero(cargar_modelo(config["modelo_comparacion"]).metadata["parametros_entrenables"])})
if filas:
    st.table(pd.DataFrame(filas).set_index("Modelo"))
replica = (f" (AUC {numero(auc_replica, 3)})" if auc_replica is not None else "")
st.markdown(f"""
Ambos modelos se compararon en igualdad de condiciones: los mismos datos, el mismo preprocesamiento y
la misma prueba final. La red neuronal **no discriminó mejor** que la regresión logística, cambia de
un entrenamiento a otro y usa más de cien veces más parámetros. Además, sus puntos no se pueden
repartir exactamente por característica. Por eso el simulador usa la regresión logística; la red se
puede elegir en el menú lateral para comparar.

La regresión logística de referencia es la del análisis del equipo (Primer_Punto.ipynb). El simulador
usa una réplica entrenada con nuestro propio código, que da prácticamente los mismos resultados{replica}.

El **AUC** es la probabilidad de que el modelo le asigne más riesgo a un mal pagador que a un buen
pagador escogidos al azar (0,5 = azar, 1 = perfecto). Con solo la información de la solicitud, los
valores observados son modestos: el riesgo de crédito es difícil de predecir.
""")

figuras = [("pd_roc.png", "Curva ROC en la prueba final"), ("pd_calibracion.png", "Calibración: riesgo predicho vs. observado"),
           ("pd_importancia.png", "Características que más pesan en la red neuronal")]
columnas = st.columns(3)
for col, (archivo, titulo) in zip(columnas, figuras):
    ruta = RAIZ / "reports/figures" / archivo
    if ruta.exists():
        col.image(str(ruta), caption=titulo)

st.markdown(f"""
### 5. Limitaciones
- Solo aplica a **créditos a 36 meses** y a perfiles parecidos a los de Lending Club entre 2007 y 2012.
- La relación deuda / ingreso se limita a **30 %**, el máximo que el modelo vio al aprender.
- El modelo aprendió con una tasa de malos de {porcentaje(tasa_train)} y la prueba final tuvo {porcentaje(tasa_test)}:
  tiende a **subestimar levemente** el riesgo de épocas posteriores.
- Es un **proyecto académico**: no es una oferta ni una decisión de crédito.
""")
