"""
Lógica de la aplicación web, separada de la interfaz para poder probarla con pytest.

Carga un modelo guardado por scripts/parte1_pd/entrenar_pd.py (no reentrena) y, para un solicitante:
    - arma su fila de entrada con las mismas columnas del entrenamiento,
    - calcula PD, puntaje y percentil frente a la población,
    - reparte el puntaje en puntos por variable (src/scorecard.py),
    - recalcula el puntaje si cambia una característica ("¿y si...?").
"""

import configparser
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.preprocesamiento import INDICADOR_DE  # noqa: E402
from src.red_neuronal import RedNeuronal  # noqa: E402
from src.scorecard import Scorecard, percentil, puntos_por_variable  # noqa: E402
from src.variables import NOMBRES, PROPOSITOS, VIVIENDA  # noqa: E402


@dataclass(frozen=True)
class Campo:
    """Cómo se pide una variable en el formulario y en el simulador."""
    etiqueta: str
    ayuda: str
    minimo: float = 0
    maximo: float = 1
    paso: float = 1
    defecto: float = 0
    opciones: dict | None = None       # categóricas: valor del modelo -> texto
    unidad: str = ""
    tope: str = ""                     # "240.000 USD": desde ahí el modelo no distingue valores mayores


# Rangos dentro de lo visto en entrenamiento (metadata["rangos"]); dti se limita a 29.99
# porque el modelo nunca vio valores mayores y la red los extrapola mal.
CAMPOS = {
    "loan_amnt": Campo("Monto que necesitas", "Monto del préstamo solicitado, en dólares.",
                       500, 35000, 500, 10000, unidad="USD"),
    "purpose": Campo("¿Para qué es el crédito?", "Finalidad declarada del préstamo.",
                     defecto="debt_consolidation", opciones=PROPOSITOS),
    "annual_inc": Campo("Ingreso anual", "Ingreso anual que declaras, en dólares.",
                        2000, 1_000_000, 1000, 56000, unidad="USD"),
    "emp_length": Campo("Años en tu empleo actual", "0 = menos de un año; 10 = diez años o más.",
                        0, 10, 1, 5, unidad="años"),
    "home_ownership": Campo("Tu vivienda", "Situación de vivienda.", defecto="RENT", opciones=VIVIENDA),
    "dti": Campo("Deuda mensual / ingreso mensual (%)",
                 "De cada 100 dólares que ganas al mes, cuántos van a pagar deudas "
                 "(sin hipoteca ni este crédito). El modelo se entrenó con valores hasta 30 %.",
                 0, 29.9, 0.1, 13, unidad="%"),
    "revol_bal": Campo("Saldo en tarjetas y cupos rotativos", "Lo que debes hoy en tarjetas y líneas rotativas.",
                       0, 300_000, 500, 8000, unidad="USD"),
    "revol_util": Campo("Uso de tu cupo rotativo (%)", "Qué porcentaje de tu cupo total de tarjetas estás usando.",
                        0, 100, 1, 48, unidad="%"),
    "open_acc": Campo("Créditos abiertos hoy", "Número de líneas de crédito abiertas.", 1, 40, 1, 9),
    "total_acc": Campo("Créditos en todo tu historial", "Total de líneas de crédito en tu historial.", 1, 90, 1, 20),
    "antig_credito_meses": Campo("Años desde tu primer crédito", "Antigüedad de tu línea de crédito más antigua.",
                                 1, 60, 1, 12, unidad="años"),
    "inq_last_6mths": Campo("Consultas a tu historial en 6 meses",
                            "Cuántas veces entidades consultaron tu historial (sin créditos de vivienda o vehículo).",
                            0, 10, 1, 1),
    "delinq_2yrs": Campo("Moras de 30+ días en los últimos 2 años", "Episodios de atraso de 30 días o más.",
                         0, 10, 1, 0),
    "mths_since_last_delinq": Campo("Meses desde tu última mora", "Si nunca has estado en mora, déjalo vacío.",
                                    0, 120, 1, 36, unidad="meses"),
    "pub_rec": Campo("Registros públicos negativos", "Quiebras, embargos o sentencias en tu contra.",
                     defecto=0, opciones={0: "Ninguno", 1: "Uno", 2: "Dos o más"}),
}

# Bandas por percentil de la población, de mejor a peor. El color de estado va siempre con
# ícono (Material Symbols) y texto, nunca solo.
BANDAS = [
    (80, "Excelente", "#067a06", "verified"),
    (60, "Bueno", "#0ca30c", "check_circle"),
    (40, "Promedio", "#c98500", "remove_circle"),
    (20, "Por mejorar", "#ec835a", "trending_down"),
    (0, "Riesgo alto", "#d03b3b", "error"),
]


def leer_config_app(ruta=RAIZ / "config.ini"):
    lector = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    lector.read(ruta, encoding="utf-8")
    seccion = lector["app"] if lector.has_section("app") else {}
    return {
        "modelo": seccion.get("modelo", "models/pd_logistica"),
        "modelo_comparacion": seccion.get("modelo_comparacion", "models/pd"),
        "enlace_reporte": seccion.get("enlace_reporte", "").strip(),
        "enlace_video": seccion.get("enlace_video", "").strip(),
        "enlace_repositorio": seccion.get("enlace_repositorio", "").strip(),
    }


def banda(pct):
    return next(b for b in BANDAS if pct >= b[0])


class Modelo:
    def __init__(self, carpeta):
        carpeta = Path(carpeta)
        if not carpeta.is_absolute():
            carpeta = RAIZ / carpeta
        self.metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
        self.red = RedNeuronal.cargar(carpeta / "red.npz")
        self.preprocesador = joblib.load(carpeta / "preprocesador.joblib")
        self.scorecard = Scorecard(**self.metadata["config"]["scorecard"])
        self.poblacion = np.sort(np.load(carpeta / "puntajes_poblacion.npy"))
        self.referencia = np.array(self.metadata["referencia_transformada"])
        capas = self.metadata["config"]["red"]["capas_ocultas"]
        self.es_logistica = not capas
        self.nombre = "Regresión logística" if self.es_logistica else "Red neuronal " + "-".join(map(str, capas))

    @property
    def columnas(self):
        return list(self.preprocesador.feature_names_in_)

    def filas(self, entradas):
        """DataFrame con las columnas del modelo a partir de uno o varios diccionarios.
        Un valor vacío (None/NaN) en antigüedad laboral o en meses desde la última mora
        enciende su indicadora, igual que en la limpieza de los datos."""
        df = pd.DataFrame([entradas] if isinstance(entradas, dict) else list(entradas))
        for indicador, variable in INDICADOR_DE.items():
            df[variable] = pd.to_numeric(df[variable], errors="coerce")
            df[indicador] = df[variable].isna().astype(int)
        return df[self.columnas]

    def evaluar(self, entradas):
        X = self.preprocesador.transform(self.filas(entradas))
        pd_est = self.red.predecir(X)
        puntaje = self.scorecard.puntaje(pd_est)
        pct = np.array([percentil(s, self.poblacion) for s in puntaje])
        return pd.DataFrame({"pd": pd_est, "puntaje": puntaje, "percentil": pct})

    def puntos(self, entradas):
        """Puntos de cada variable frente al solicitante promedio, de mayor a menor efecto."""
        X = self.preprocesador.transform(self.filas(entradas))
        base, puntos, interaccion = puntos_por_variable(self.red, self.scorecard, X, self.referencia,
                                                        self.preprocesador.grupos())
        serie = puntos.iloc[0].rename(index=NOMBRES)
        return base, serie.reindex(serie.abs().sort_values(ascending=False).index), float(interaccion[0])

    def barrido(self, entradas, variable, valores):
        """Puntaje del solicitante si la variable tomara cada uno de los valores."""
        filas = [{**entradas, variable: v} for v in valores]
        return self.evaluar(filas)["puntaje"].to_numpy()

    def cortes_bandas(self):
        """Puntaje donde empieza cada banda (percentiles 20, 40, 60 y 80 de la población) y
        la escala del velocímetro (percentiles 0,5 y 99,5, redondeados a la decena)."""
        cortes = {nombre: float(np.percentile(self.poblacion, p)) for p, nombre, *_ in BANDAS if p > 0}
        minimo = 10 * np.floor(np.percentile(self.poblacion, 0.5) / 10)
        maximo = 10 * np.ceil(np.percentile(self.poblacion, 99.5) / 10)
        return cortes, float(minimo), float(maximo)

    def resumen_poblacion(self):
        return {"n": len(self.poblacion), "mediana": float(np.median(self.poblacion)),
                "p05": float(np.percentile(self.poblacion, 5)), "p95": float(np.percentile(self.poblacion, 95))}


def campos_del_modelo(modelo):
    """CAMPOS con el máximo de cada control en el tope que aprendió el modelo.

    El preprocesador recorta varias variables en el percentil 99 de train (topes_): para el
    modelo, 2 moras y 10 moras son lo mismo. Ofrecer valores por encima del tope haría que el
    puntaje no cambie y el simulador mostrara una curva plana, así que el máximo del control
    queda en el tope ("tope o más"). Se lee del modelo guardado: si se reentrena, se ajusta solo.
    Los conteos con tope pequeño (p. ej. moras: 0, 1, "2 o más") se vuelven una lista.
    """
    campos = dict(CAMPOS)
    for var, tope in modelo.preprocesador.topes_.items():
        c = campos[var]
        escala = 12 if var == "antig_credito_meses" else 1
        maximo = math.ceil(tope / escala / c.paso) * c.paso
        if maximo >= c.maximo:
            continue
        texto = f"{maximo:,.0f}".replace(",", ".") + (f" {c.unidad}" if c.unidad else "")
        ayuda = f"{c.ayuda} El modelo trata igual cualquier valor desde {texto}, por eso el máximo es «{texto} o más»."
        if maximo <= 3 and float(c.paso).is_integer():
            opciones = {k: str(k) for k in range(int(maximo))} | {int(maximo): f"{int(maximo)} o más"}
            campos[var] = replace(c, maximo=maximo, defecto=min(c.defecto, maximo), opciones=opciones, ayuda=ayuda)
        else:
            campos[var] = replace(c, maximo=maximo, defecto=min(c.defecto, maximo), ayuda=ayuda, tope=texto)
    return campos


def solicitante_por_defecto():
    """Valores iniciales del formulario: un solicitante típico de entrenamiento."""
    valores = {k: c.defecto for k, c in CAMPOS.items()}
    valores["antig_credito_meses"] = CAMPOS["antig_credito_meses"].defecto * 12
    valores["mths_since_last_delinq"] = None          # nunca ha estado en mora (lo más frecuente)
    return valores


def _candidatos(variable, actual, campo):
    """Un cambio realista por variable, en la dirección que reduce el riesgo y dentro del rango
    del control (no se sugiere, por ejemplo, pasar de usar el 85 % del cupo al 10 %)."""
    if variable == "revol_util":            # meta habitual: usar menos del 30 % del cupo
        return [30] if actual > 30 else ([10] if actual > 10 else [])
    if variable == "inq_last_6mths":
        return [0] if actual > 0 else []
    if variable == "dti":
        return [round(actual - 5, 1)] if actual >= 5 else []
    if variable == "revol_bal":
        return [round(actual / 2, -2)] if actual >= 1000 else []
    if variable == "loan_amnt":
        return [v for v in (round(actual * 0.75 / 500) * 500,) if campo.minimo <= v < actual]
    return []


# Lo que una persona puede cambiar de verdad antes de pedir el crédito (no su historial pasado)
ACCIONABLES = {
    "revol_util": "Usar menos tu cupo rotativo: de {a} a {n}",
    "inq_last_6mths": "Evitar nuevas consultas a tu historial: de {a} a {n}",
    "dti": "Reducir tu deuda mensual frente a tu ingreso: de {a} a {n}",
    "revol_bal": "Pagar parte de tus tarjetas: saldo de {a} a {n}",
    "loan_amnt": "Pedir un monto menor: de {a} a {n}",
}


def mejoras(modelo, entradas, campos, formatear, maximo=3, minimo_puntos=1.0):
    """Los cambios accionables que más suben el puntaje, calculados con el mismo modelo.

    Para cada variable accionable se prueba un valor realista (p. ej. uso del cupo al 30 %)
    dejando todo lo demás igual.
    Devuelve una lista de (texto, puntos ganados, puntaje nuevo), de mayor a menor ganancia.
    formatear(variable, valor) -> texto del valor con su unidad.
    """
    actual = float(modelo.evaluar(entradas)["puntaje"].iloc[0])
    resultado = []
    for variable, plantilla in ACCIONABLES.items():
        valores = _candidatos(variable, entradas[variable], campos[variable])
        if not valores:
            continue
        puntajes = modelo.barrido(entradas, variable, valores)
        mejor = int(np.argmax(puntajes))
        ganancia = float(puntajes[mejor]) - actual
        if ganancia >= minimo_puntos:
            texto = plantilla.format(a=formatear(variable, entradas[variable]), n=formatear(variable, valores[mejor]))
            resultado.append((texto, ganancia, float(puntajes[mejor])))
    return sorted(resultado, key=lambda r: -r[1])[:maximo]
