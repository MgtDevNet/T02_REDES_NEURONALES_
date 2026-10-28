"""Estilo visual y piezas compartidas por las páginas de la aplicación.

Paleta morada y tipografía Inter (libre, Google Fonts). Íconos monocromáticos de Material
Symbols: en Streamlit se escriben como :material/nombre: y en el HTML propio con la clase "icono".
"""

from urllib.parse import quote

import streamlit as st

from motor import Modelo, leer_config_app

OSCURO = "#3c096c"       # fondo del encabezado
PRIMARIO = "#7b2cbf"
TINTA = "#2d0a4e"        # textos y marcas sobre los gráficos
TINTA_SUAVE = "#5b5470"
# Colores de los gráficos, validados para daltonismo (scripts/validate_palette.js de la guía):
SUMA, RESTA = "#7b2cbf", "#e34948"          # par divergente: morado suma puntos, rojo resta
ACTUAL, NUEVO = "#7b2cbf", "#1baf7a"        # simulador: tu situación vs. el nuevo valor
NEUTRO = "#d9d4e2"                          # opciones que no estás comparando
POBLACION = "#c9b0ec"                       # histograma de la población (una sola serie)



def _ilustracion_encabezado():
    """Ilustración abstracta propia (SVG): líneas que fluyen y círculos translúcidos en blanco,
    para superponer al degradado morado del encabezado. Se incrusta como data URI: no es un
    archivo aparte ni tiene licencia de terceros."""
    curvas = []
    for i in range(16):
        y0, y1, a = 40 + i * 13, 10 + i * 17, 70 + i * 4
        opacidad = 0.05 + 0.012 * i
        curvas.append(f'<path d="M380,{y0} C620,{y0 - a} 820,{y1 + a} 1200,{y1}" fill="none" '
                      f'stroke="white" stroke-width="1.2" stroke-opacity="{opacidad:.3f}"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 260" preserveAspectRatio="xMaxYMid slice">'
           '<circle cx="1060" cy="30" r="150" fill="white" fill-opacity="0.06"/>'
           '<circle cx="1170" cy="240" r="95" fill="white" fill-opacity="0.05"/>'
           '<circle cx="880" cy="250" r="40" fill="none" stroke="white" stroke-opacity="0.18" stroke-width="1.5"/>'
           + "".join(curvas) + "</svg>")
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


ILUSTRACION = _ilustracion_encabezado()


def _decoracion_fondo(anillos, con_ondas):
    """Anillos concéntricos y, si se pide, líneas en onda en morado muy tenue, para las esquinas
    del fondo de la página. Repiten el motivo del encabezado sin competir con el contenido."""
    figuras = [f'<circle cx="400" cy="400" r="{60 + 52 * i}" fill="none" stroke="{PRIMARIO}" '
               f'stroke-opacity="{0.13 - 0.012 * i:.3f}" stroke-width="1.3"/>' for i in range(anillos)]
    if con_ondas:
        for i in range(10):
            y = 470 + i * 26
            figuras.append(f'<path d="M0,{y} C220,{y - 90} 470,{y + 90} 800,{y - 40}" fill="none" '
                           f'stroke="{PRIMARIO}" stroke-opacity="{0.05 + 0.008 * i:.3f}" stroke-width="1.2"/>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">' + "".join(figuras) + "</svg>")
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


FONDO_ESQUINA = _decoracion_fondo(anillos=7, con_ondas=True)
FONDO_ANILLOS = _decoracion_fondo(anillos=4, con_ondas=False)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=swap');

html, body, .stApp, p, li, label, input, textarea, button, td, th, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stCaption, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
h1, h2, h3, h4, h5, h6, .puntaje-valor,
[data-testid="stHeadingWithActionElements"], [data-testid="stHeadingWithActionElements"] * {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}}
/* los íconos dentro de los títulos conservan su fuente de símbolos; en markdown, Streamlit
   dibuja :material/...: como <span role="img" translate="no"> con la fuente en línea */
[data-testid="stIconMaterial"], [data-testid="stHeadingWithActionElements"] [data-testid="stIconMaterial"],
[data-testid="stHeadingWithActionElements"] span[role="img"], span[role="img"][translate="no"],
.icono {{ font-family: 'Material Symbols Rounded' !important; }}
h1, h2, h3, h4 {{ color: {TINTA}; font-weight: 700; letter-spacing: -0.3px; }}
/* Fondo de la página: lila claro con dos brillos morados muy tenues en las esquinas */
.stApp, [data-testid="stAppViewContainer"] {{
    background:
        url("{FONDO_ESQUINA}") right -180px bottom -220px / 760px 760px no-repeat,
        url("{FONDO_ANILLOS}") left -200px top 160px / 520px 520px no-repeat,
        radial-gradient(rgba(123, 44, 191, .10) 1px, transparent 1.4px) 0 0 / 24px 24px,
        radial-gradient(1100px 520px at 105% -8%, rgba(123, 44, 191, .14), transparent 62%),
        radial-gradient(900px 480px at -8% 108%, rgba(60, 9, 108, .10), transparent 60%),
        #f7f4fb;
    background-attachment: fixed;
}}
header[data-testid="stHeader"], [data-testid="stHeader"] {{ background: transparent; }}
/* Panel lateral: el mismo lenguaje del fondo. Degradado lila, puntos y anillos/ondas solo en la
   parte baja, para no quedar detrás del menú ni de los controles. */
section[data-testid="stSidebar"] {{
    background:
        url("{FONDO_ESQUINA}") left -300px bottom -330px / 640px 640px no-repeat,
        radial-gradient(rgba(123, 44, 191, .09) 1px, transparent 1.4px) 0 0 / 22px 22px,
        linear-gradient(180deg, #ffffff 0%, #f6f1fd 45%, #ece1fa 100%);
    border-right: 1px solid #e7e1f0;
}}
[data-testid="stSidebarContent"], [data-testid="stSidebarHeader"],
[data-testid="stSidebarUserContent"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
.icono {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
          font-size: 1.15em; line-height: 1; vertical-align: -0.2em; display: inline-block;
          font-variation-settings: 'FILL' 1; -webkit-font-smoothing: antialiased; }}
.banda-superior {{
    background: url("{ILUSTRACION}") right center / cover no-repeat,
                linear-gradient(120deg, {OSCURO} 0%, #5a189a 55%, {PRIMARIO} 100%);
    color: #fff; border-radius: 18px; padding: 30px 32px; margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(60, 9, 108, .18);
}}
.banda-superior h1 {{ color: #fff; font-size: 2rem; font-weight: 800; margin: 0; }}
.banda-superior p {{ color: #eadcfa; margin: 8px 0 0 0; font-size: 1.02rem; }}
.marca {{ font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #d4b8f5; }}
.tarjeta {{
    background: #fff; border: 1px solid #e7e1f0; border-radius: 18px;
    padding: 22px 24px; box-shadow: 0 2px 10px rgba(60, 9, 108, .06);
}}
.puntaje-etiqueta {{ color: {TINTA_SUAVE}; font-size: .82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
.puntaje-valor {{ color: {TINTA}; font-size: 4.2rem; font-weight: 800; line-height: 1.05; letter-spacing: -2px; }}
.chip {{ display: inline-block; padding: 5px 14px 5px 10px; border-radius: 999px; font-weight: 600;
         font-size: .95rem; margin-top: 6px; }}
.dato {{ color: {TINTA_SUAVE}; font-size: .85rem; margin-top: 16px; }}
.dato b {{ color: {TINTA}; font-size: 1.4rem; font-weight: 700; display: block; }}
.aviso {{ color: {TINTA_SUAVE}; font-size: .8rem; }}
.lema {{ text-align: center; color: {PRIMARIO}; font-size: 1.2rem; font-weight: 700; margin: 4px 0 14px 0; }}
.lema b {{ color: {OSCURO}; }}
/* Franja de pasos bajo el encabezado */
.pasos {{ display: flex; gap: 14px; margin: -4px 0 22px 0; flex-wrap: wrap; }}
.paso {{ flex: 1 1 220px; display: flex; gap: 12px; align-items: center; background: #ffffffcc;
         border: 1px solid #e7e1f0; border-radius: 14px; padding: 12px 16px; }}
.paso .icono {{ color: {PRIMARIO}; font-size: 1.7rem; background: #f1e8fc; border-radius: 10px; padding: 6px; }}
.paso b {{ color: {TINTA}; display: block; font-size: .95rem; }}
.paso span.txt {{ color: {TINTA_SUAVE}; font-size: .82rem; }}
/* Sugerencias de mejora */
.mejora {{ display: flex; justify-content: space-between; align-items: center; gap: 12px;
           background: #fff; border: 1px solid #e7e1f0; border-left: 4px solid {PRIMARIO};
           border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; }}
.mejora-texto {{ color: {TINTA}; font-size: .92rem; }}
.mejora-puntos {{ color: #067a06; font-weight: 700; white-space: nowrap; background: #e7f6e7;
                  border-radius: 999px; padding: 3px 10px; font-size: .9rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 18px; }}
</style>
"""


def aplicar_estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def icono(nombre):
    """Ícono Material Symbols para usar dentro de HTML propio."""
    return f'<span class="icono">{nombre}</span>'


def encabezado(titulo, subtitulo):
    st.markdown(f"""<div class="banda-superior"><div class="marca">¿Me prestas? · Simulador de crédito</div>
        <h1>{titulo}</h1><p>{subtitulo}</p></div>""", unsafe_allow_html=True)


def pasos(lista):
    """Franja de pasos: [(ícono Material, título, texto), ...]."""
    st.markdown('<div class="pasos">' + "".join(
        f'<div class="paso">{icono(simbolo)}<div><b>{titulo}</b><span class="txt">{texto}</span></div></div>'
        for simbolo, titulo, texto in lista) + "</div>", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(carpeta):
    return Modelo(carpeta)


def barra_lateral():
    """Selector de modelo y enlaces. Devuelve (modelo, configuración de la app)."""
    config = leer_config_app()
    with st.sidebar:
        st.markdown("### :material/account_balance: ¿Me prestas?")
        st.caption("Estimación del riesgo de incumplimiento para créditos de consumo a 36 meses.")
        opciones = {"Regresión logística (recomendado)": config["modelo"],
                    "Red neuronal (comparación)": config["modelo_comparacion"]}
        eleccion = st.radio("Modelo", list(opciones), key="modelo_elegido",
                            help="La regresión logística tuvo igual o mejor desempeño que la red en "
                                 "datos que ninguno de los dos vio, y sus puntos se explican variable "
                                 "por variable. Ver «Cómo funciona».")
        modelo = cargar_modelo(opciones[eleccion])
        st.divider()
        st.markdown("**Enlaces**")
        for texto, clave, simbolo in (("Reporte técnico", "enlace_reporte", ":material/description:"),
                                      ("Video promocional", "enlace_video", ":material/smart_display:"),
                                      ("Repositorio", "enlace_repositorio", ":material/code:")):
            if config[clave]:
                st.link_button(texto, config[clave], icon=simbolo, width="stretch")
            else:
                st.caption(f"{simbolo} {texto}: disponible pronto")
        st.divider()
        st.markdown('<p class="aviso">Proyecto académico (Universidad Nacional de Colombia, 2026-02). '
                    'Basado en datos históricos de Lending Club 2007–2012. No es una decisión de crédito real.</p>',
                    unsafe_allow_html=True)
    return modelo, config
