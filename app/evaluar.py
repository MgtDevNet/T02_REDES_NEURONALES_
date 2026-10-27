"""Página principal: el solicitante ingresa sus datos y obtiene puntaje, PD, percentil,
qué variables explican su puntaje y cómo cambiaría si cambia alguna característica."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from estilo import (ACTUAL, NEUTRO, NUEVO, POBLACION, RESTA, SUMA, TINTA, TINTA_SUAVE, aplicar_estilo,
                    barra_lateral, encabezado, icono, pasos)
from formato import numero, porcentaje, puntos_porcentuales, usd
from graficos import velocimetro
from motor import banda, campos_del_modelo, mejoras

aplicar_estilo()
modelo, config = barra_lateral()
CAMPOS = campos_del_modelo(modelo)       # máximos en el tope que aprendió el modelo
encabezado("¿Qué tan probable es que te presten?",
           "Ingresa tus datos y conoce tu puntaje de crédito, tu probabilidad de incumplimiento, "
           "cómo te ubicas frente a otros solicitantes y qué podrías mejorar.")
pasos([("edit_note", "1. Ingresa tus datos", "Tu crédito, tus ingresos y tu historial."),
       ("speed", "2. Conoce tu puntaje", "Tu riesgo y cómo te ubicas frente a otros."),
       ("tune", "3. Simula mejoras", "Mira qué cambios suben tu puntaje.")])

EMPLEO = {None: "Prefiero no decir", 0: "Menos de 1 año", **{k: f"{k} años" for k in range(1, 10)},
          10: "10 años o más"}


def limites(c, valor=None):
    """Mínimo, máximo, paso y valor del mismo tipo (Streamlit no admite mezclar int y float)."""
    valor = c.defecto if valor is None else valor
    tipo = int if all(float(v).is_integer() for v in (c.minimo, c.maximo, c.paso)) else float
    return dict(min_value=tipo(c.minimo), max_value=tipo(c.maximo), step=tipo(c.paso), value=tipo(valor))


def rotulo(c, con_unidad=False):
    texto = f"{c.etiqueta} ({c.unidad})" if con_unidad and c.unidad else c.etiqueta
    return f"{texto} · máx.: {c.tope} o más" if c.tope else texto


def campo_numerico(var, col=st):
    c = CAMPOS[var]
    return col.number_input(rotulo(c, con_unidad=True), help=c.ayuda, key=var, **limites(c))


def deslizador(var, col=st):
    c = CAMPOS[var]
    return col.slider(rotulo(c), help=c.ayuda, key=var, **limites(c))


def lista(var, col=st):
    c = CAMPOS[var]
    claves = list(c.opciones)
    return col.selectbox(c.etiqueta, claves, index=claves.index(c.defecto),
                         format_func=c.opciones.get, help=c.ayuda, key=var)


# -----------------------------------------------------------------------------
# Formulario
# -----------------------------------------------------------------------------
izquierda, derecha = st.columns([1.15, 1], gap="large")

with izquierda:
    with st.container(border=True):
        st.markdown("#### :material/credit_card: El crédito")
        a, b = st.columns(2)
        loan_amnt = campo_numerico("loan_amnt", a)
        purpose = lista("purpose", b)
        st.caption("Plazo: 36 meses. El modelo se construyó solo con créditos a 36 meses.")

    with st.container(border=True):
        st.markdown("#### :material/person: Sobre ti")
        a, b = st.columns(2)
        annual_inc = campo_numerico("annual_inc", a)
        emp_length = a.selectbox(CAMPOS["emp_length"].etiqueta, list(EMPLEO), index=6,
                                 format_func=EMPLEO.get, help=CAMPOS["emp_length"].ayuda, key="emp_length")
        home_ownership = lista("home_ownership", b)
        dti = deslizador("dti", b)

    with st.container(border=True):
        st.markdown("#### :material/monitoring: Tu historial crediticio")
        a, b = st.columns(2)
        revol_bal = campo_numerico("revol_bal", a)
        revol_util = deslizador("revol_util", a) if revol_bal > 0 else 0.0
        if revol_bal == 0:
            a.caption("Sin saldo rotativo, el uso de tu cupo es 0 %.")
        open_acc = campo_numerico("open_acc", a)
        total_acc = campo_numerico("total_acc", a)
        anios_credito = deslizador("antig_credito_meses", b)
        inq_last_6mths = deslizador("inq_last_6mths", b)
        delinq_2yrs = (lista if CAMPOS["delinq_2yrs"].opciones else campo_numerico)("delinq_2yrs", b)
        tuvo_mora = b.toggle("He tenido alguna mora en mi historial", key="tuvo_mora")
        mths_since_last_delinq = campo_numerico("mths_since_last_delinq", b) if tuvo_mora else None
        pub_rec = lista("pub_rec", b)

        if total_acc < open_acc:
            st.warning("Los créditos de todo tu historial no pueden ser menos que los abiertos hoy: "
                       "se usa el número de créditos abiertos.")
            total_acc = open_acc
        if tuvo_mora and mths_since_last_delinq is not None and mths_since_last_delinq <= 24 and delinq_2yrs == 0:
            st.info("Si tu última mora fue hace 2 años o menos, las moras en los últimos 2 años deberían ser al menos 1.")

entradas = {
    "loan_amnt": loan_amnt, "purpose": purpose, "annual_inc": annual_inc, "emp_length": emp_length,
    "home_ownership": home_ownership, "dti": dti, "revol_bal": revol_bal, "revol_util": revol_util,
    "open_acc": open_acc, "total_acc": total_acc, "antig_credito_meses": anios_credito * 12,
    "inq_last_6mths": inq_last_6mths, "delinq_2yrs": delinq_2yrs,
    "mths_since_last_delinq": mths_since_last_delinq, "pub_rec": pub_rec,
}
resultado = modelo.evaluar(entradas).iloc[0]
puntaje, pd_est, pct = resultado["puntaje"], resultado["pd"], resultado["percentil"]
_, nombre_banda, color_banda, simbolo = banda(pct)

# -----------------------------------------------------------------------------
# Resultado
# -----------------------------------------------------------------------------
with derecha:
    cortes, _, _ = modelo.cortes_bandas()
    st.markdown(f"""<div class="tarjeta">
        <div class="puntaje-etiqueta">Tu puntaje de crédito</div>
        {velocimetro(puntaje, pct, cortes, TINTA, TINTA_SUAVE)}
        <div style="text-align:center"><span class="chip" style="background:{color_banda}1f; border:1px solid {color_banda}; color:{TINTA}"><span style="color:{color_banda}">{icono(simbolo)}</span> {nombre_banda}</span></div>
        <div style="display:flex; gap:28px; justify-content:center; text-align:center">
          <div class="dato"><b>{porcentaje(pd_est)}</b>probabilidad de no pagar</div>
          <div class="dato"><b>{numero(pct)} %</b>de los solicitantes tiene<br>un puntaje igual o menor</div>
        </div></div>""", unsafe_allow_html=True)

    def formatear(variable, valor):
        if variable in ("revol_bal", "loan_amnt"):
            return usd(valor)
        if variable in ("revol_util", "dti"):
            return f"{numero(valor, 1 if variable == 'dti' else 0)} %"
        return numero(valor)

    sugerencias = mejoras(modelo, entradas, CAMPOS, formatear)
    st.markdown("##### :material/trending_up: Lo que más te ayudaría")
    if sugerencias:
        st.markdown("".join(
            f'<div class="mejora"><span class="mejora-texto">{texto}</span>'
            f'<span class="mejora-puntos">{numero(ganancia, 1, signo=True)} pts</span></div>'
            for texto, ganancia, _ in sugerencias), unsafe_allow_html=True)
        st.caption("Calculado con el mismo modelo, cambiando una sola cosa a la vez. Es una estimación, no una garantía.")
    else:
        st.caption("Con tus datos actuales no encontramos cambios sencillos que suban tu puntaje de forma notable.")

    st.markdown("##### :material/groups: Cómo te ubicas frente a otros solicitantes")
    conteos, bordes = np.histogram(modelo.poblacion, bins=45)
    hist = pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "créditos": conteos})
    barras = alt.Chart(hist).mark_bar(color=POBLACION, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("desde:Q", title="Puntaje", scale=alt.Scale(zero=False)), x2="hasta:Q",
        y=alt.Y("créditos:Q", title="Solicitantes"),
        tooltip=[alt.Tooltip("desde:Q", format=".0f", title="Desde"),
                 alt.Tooltip("hasta:Q", format=".0f", title="Hasta"), "créditos:Q"])
    marca = pd.DataFrame({"puntaje": [puntaje], "texto": [f"Tú: {numero(puntaje)}"]})
    regla = alt.Chart(marca).mark_rule(color=TINTA, strokeWidth=2.5).encode(x="puntaje:Q")
    texto = alt.Chart(marca).mark_text(align="left", dx=6, dy=-6, color=TINTA, fontWeight="bold").encode(
        x="puntaje:Q", y=alt.value(12), text="texto:N")
    st.altair_chart((barras + regla + texto).properties(height=210), width="stretch")
    st.caption(f"{numero(len(modelo.poblacion))} créditos a 36 meses de Lending Club (2007–2012). "
               "Más puntaje = menos riesgo.")

# -----------------------------------------------------------------------------
# Qué explica el puntaje
# -----------------------------------------------------------------------------
st.divider()
explica, simula = st.columns(2, gap="large")

with explica:
    st.markdown("### :material/insights: ¿Qué explica tu puntaje?")
    base, puntos, interaccion = modelo.puntos(entradas)
    st.caption(f"Puntos que te suma o te resta cada característica frente a un solicitante promedio "
               f"({numero(base)} puntos).")
    top = puntos.head(8).rename("puntos").reset_index().rename(columns={"index": "variable"})
    top["efecto"] = np.where(top["puntos"] >= 0, "Te suma", "Te resta")
    top["etiqueta"] = top["puntos"].map(lambda v: numero(v, 1, signo=True))
    orden = list(top["variable"])
    grafico = alt.Chart(top).encode(
        y=alt.Y("variable:N", sort=orden, title=None),
        x=alt.X("puntos:Q", title="Puntos"),
        tooltip=["variable:N", alt.Tooltip("puntos:Q", format="+.1f")])
    barras = grafico.mark_bar(cornerRadiusEnd=3, height=16).encode(
        color=alt.Color("efecto:N", scale=alt.Scale(domain=["Te suma", "Te resta"], range=[SUMA, RESTA]),
                        legend=alt.Legend(title=None, orient="top")))
    positivas = grafico.transform_filter("datum.puntos >= 0").mark_text(align="left", dx=4, color=TINTA)
    negativas = grafico.transform_filter("datum.puntos < 0").mark_text(align="right", dx=-4, color=TINTA)
    etiquetas = positivas.encode(text="etiqueta:N") + negativas.encode(text="etiqueta:N")
    st.altair_chart((barras + etiquetas).properties(height=290), width="stretch")
    if modelo.es_logistica:
        st.caption(f"Con la regresión logística los puntos suman exactamente tu puntaje: "
                   f"{numero(base)} + ({numero(puntos.sum(), 1, signo=True)}) = {numero(puntaje)}.")
    else:
        st.caption(f"En la red neuronal los puntos no suman exactamente: {numero(interaccion, 1, signo=True)} puntos vienen de "
                   "combinaciones entre variables que no se pueden atribuir a una sola.")

# -----------------------------------------------------------------------------
# ¿Y si...?
# -----------------------------------------------------------------------------
SIMULABLES = ["revol_util", "inq_last_6mths", "annual_inc", "loan_amnt", "dti", "purpose", "revol_bal",
              "open_acc", "antig_credito_meses", "delinq_2yrs", "home_ownership", "pub_rec"]

with simula:
    st.markdown("### :material/tune: ¿Y si cambias algo?")
    st.caption("Elige una característica y mira cómo cambiaría tu puntaje, dejando todo lo demás igual.")
    var = st.selectbox("Característica", SIMULABLES, format_func=lambda v: CAMPOS[v].etiqueta, key="var_simulada")
    c = CAMPOS[var]
    actual = entradas[var] / 12 if var == "antig_credito_meses" else entradas[var]
    a_modelo = (lambda v: v * 12) if var == "antig_credito_meses" else (lambda v: v)

    if c.opciones:
        valores = list(c.opciones)
        nuevo = st.selectbox("Nuevo valor", valores, index=valores.index(actual), format_func=c.opciones.get,
                             key=f"nuevo_{var}")
        puntajes = modelo.barrido(entradas, var, valores)
        datos = pd.DataFrame({"valor": [c.opciones[v] for v in valores], "puntaje": puntajes,
                              "tipo": ["Tu situación" if v == actual else ("Nuevo valor" if v == nuevo else "Otros")
                                       for v in valores]})
        grafico = alt.Chart(datos).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("valor:N", sort="-x", title=None),
            x=alt.X("puntaje:Q", title="Puntaje", scale=alt.Scale(zero=False)),
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Tu situación", "Nuevo valor", "Otros"],
                                                      range=[ACTUAL, NUEVO, NEUTRO]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["valor:N", alt.Tooltip("puntaje:Q", format=".0f")])
    else:
        nuevo = st.slider("Nuevo valor", key=f"nuevo_{var}", **limites(c, actual))
        rejilla = np.linspace(c.minimo, c.maximo, 60)
        datos = pd.DataFrame({"valor": rejilla, "puntaje": modelo.barrido(entradas, var, a_modelo(rejilla))})
        puntos_marcados = pd.DataFrame({"valor": [actual, nuevo], "tipo": ["Tu situación", "Nuevo valor"],
                                        "puntaje": modelo.barrido(entradas, var, [a_modelo(actual), a_modelo(nuevo)])})
        linea = alt.Chart(datos).mark_line(color=NEUTRO, strokeWidth=3).encode(
            x=alt.X("valor:Q", title=f"{c.etiqueta}" + (f" ({c.unidad})" if c.unidad else "")),
            y=alt.Y("puntaje:Q", title="Puntaje", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("valor:Q", format=",.1f"), alt.Tooltip("puntaje:Q", format=".0f")])
        marcas = alt.Chart(puntos_marcados).mark_point(filled=True, size=110, stroke="#fff", strokeWidth=2).encode(
            x="valor:Q", y="puntaje:Q",
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Tu situación", "Nuevo valor"], range=[ACTUAL, NUEVO]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["tipo:N", alt.Tooltip("valor:Q", format=",.1f"), alt.Tooltip("puntaje:Q", format=".0f")])
        grafico = linea + marcas

    simulado = modelo.evaluar({**entradas, var: a_modelo(nuevo)}).iloc[0]
    m1, m2 = st.columns(2)
    m1.metric("Nuevo puntaje", numero(simulado["puntaje"]), f"{numero(simulado['puntaje'] - puntaje, 1, signo=True)} puntos")
    m2.metric("Nueva probabilidad de no pagar", porcentaje(simulado["pd"]),
              puntos_porcentuales(simulado["pd"] - pd_est), delta_color="inverse")
    st.altair_chart(grafico.properties(height=250), width="stretch")

st.divider()
st.markdown('<p class="lema"><b>¿Me prestas?</b>: conoce tu riesgo, entiéndelo y mejóralo</p>',
            unsafe_allow_html=True)
st.markdown('<p class="aviso">Este simulador es un proyecto académico. El puntaje se calcula con un modelo '
            f'estadístico ({modelo.nombre}) entrenado con créditos históricos y no constituye una oferta ni '
            'una decisión de crédito.</p>', unsafe_allow_html=True)
