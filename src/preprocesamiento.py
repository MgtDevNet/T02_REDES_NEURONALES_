"""
Preprocesamiento que se ajusta con el conjunto de entrenamiento y se guarda junto al modelo.

La limpieza (filtrar filas, construir y, asignar la partición, corregir tipos) se hace antes,
en el script de limpieza de datos. Aquí solo va lo que depende de estadísticos de
entrenamiento (medianas, medias, desviaciones, categorías), porque eso debe calcularse
únicamente con train y aplicarse igual en validación, prueba y en la aplicación web.

    numéricas:   imputar mediana (+ indicador de faltante)  ->  estandarizar
    logarítmicas: imputar mediana (+ indicador)  ->  log(1 + x)  ->  estandarizar
    categóricas: imputar la categoría más frecuente  ->  one-hot
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def construir_preprocesador(numericas, categoricas, logaritmicas=()):
    # add_indicator=True agrega una columna 0/1 por cada variable que tuvo faltantes en
    # train. Para mths_since_last_delinq, por ejemplo, el faltante significa "nunca ha estado
    # en mora", así que el indicador conserva esa información en lugar de perderla al imputar.
    numericas = [c for c in numericas if c not in logaritmicas]
    transformadores = []
    if numericas:
        transformadores.append(("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
        ]), list(numericas)))
    if logaritmicas:
        transformadores.append(("log", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escalar", StandardScaler()),
        ]), list(logaritmicas)))
    if categoricas:
        transformadores.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categoricas)))
    return ColumnTransformer(transformadores, verbose_feature_names_out=False)


# Indicadoras que salen de una variable original: van con ella al permutarla, al calcular
# sus puntos en la scorecard y en la app (un valor observado implica indicador 0).
INDICADOR_DE = {"nunca_mora": "mths_since_last_delinq", "emp_length_na": "emp_length"}


class PreprocesadorCredito:
    """Preprocesamiento de la Parte 1 (reglas de María Fernanda Valencia, Punto_1/).

    Todo lo que depende de los datos se aprende en fit() solo con train y se guarda, para
    aplicarlo igual en validación, prueba y en la aplicación web:

        numéricas:   faltante -> mediana de train
                     tope superior en el percentil 99 de train (solo las de `con_tope`)
                     log(1 + x) (solo las de `logaritmicas`), después del tope
                     estandarizar con media y desviación de train
        indicadoras: ya valen 0 o 1 (nunca_mora, emp_length_na); pasan sin cambios
        categóricas: las categorías con menos de `frecuencia_minima` de train, y las que no
                     aparecen en train, se agrupan en "otras"; una columna 0/1 por categoría,
                     sin la más frecuente (categoría de referencia)

    El orden imputar -> tope -> log importa: la mediana y los topes se calculan sobre la
    variable original, y el log se aplica a valores ya recortados.
    """

    OTRAS = "otras"

    def __init__(self, numericas, categoricas=(), logaritmicas=(), con_tope=(), indicadoras=(),
                 percentil_tope=0.99, frecuencia_minima=0.01):
        self.numericas = list(numericas)
        self.categoricas = list(categoricas)
        self.logaritmicas = list(logaritmicas)
        self.con_tope = list(con_tope)
        self.indicadoras = list(indicadoras)
        self.percentil_tope = percentil_tope
        self.frecuencia_minima = frecuencia_minima
        for nombre, lista in (("logaritmicas", self.logaritmicas), ("con_tope", self.con_tope)):
            fuera = set(lista) - set(self.numericas)
            if fuera:
                raise ValueError(f"{nombre} debe ser un subconjunto de numericas: {sorted(fuera)}")

    def _numericas_sin_escalar(self, X):
        Z = X[self.numericas].astype(float).fillna(self.medianas_)
        for col, tope in self.topes_.items():
            Z[col] = Z[col].clip(upper=tope)
        for col in self.logaritmicas:
            Z[col] = np.log1p(Z[col].clip(lower=0))
        return Z

    def _agrupar(self, X, col):
        s = X[col].astype(object).where(X[col].notna(), self.referencia_[col]).astype(str)
        conocidas = set(self.categorias_[col])
        destino = self.OTRAS if self.OTRAS in conocidas else self.referencia_[col]
        return s.where(s.isin(conocidas), destino)

    def fit(self, X, y=None):
        # columnas de entrada, con el mismo nombre que usa scikit-learn (scripts/parte2_lgd/estimar_perdida.py lo lee)
        self.feature_names_in_ = np.array(self.numericas + self.indicadoras + self.categoricas, dtype=object)
        self.medianas_ = X[self.numericas].astype(float).median()
        self.topes_ = X[self.con_tope].astype(float).quantile(self.percentil_tope)

        Z = self._numericas_sin_escalar(X)
        self.medias_ = Z.mean()
        desviaciones = Z.std(ddof=0)
        self.desviaciones_ = desviaciones.where(desviaciones > 0, 1.0)

        self.categorias_, self.referencia_ = {}, {}
        for col in self.categoricas:
            frecuencia = X[col].astype(str).value_counts(normalize=True)
            frecuentes = sorted(frecuencia[frecuencia >= self.frecuencia_minima].index)
            if (frecuencia < self.frecuencia_minima).any():
                frecuentes.append(self.OTRAS)
            self.categorias_[col] = frecuentes
            self.referencia_[col] = frecuencia.idxmax()
        return self

    def transform(self, X):
        Z = (self._numericas_sin_escalar(X) - self.medias_) / self.desviaciones_
        bloques = [Z.to_numpy(), X[self.indicadoras].to_numpy(dtype=float)]
        for col in self.categoricas:
            s = self._agrupar(X, col).to_numpy()
            niveles = [c for c in self.categorias_[col] if c != self.referencia_[col]]
            bloques.append(np.column_stack([s == c for c in niveles]).astype(float)
                           if niveles else np.empty((len(X), 0)))
        return np.hstack(bloques)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        nombres = self.numericas + self.indicadoras
        for col in self.categoricas:
            nombres += [f"{col}_{c}" for c in self.categorias_[col] if c != self.referencia_[col]]
        return np.array(nombres, dtype=object)

    def grupos(self):
        """Columnas transformadas de cada variable original, p. ej.
        {"purpose": [índices de sus dummies], "mths_since_last_delinq": [i, i_nunca_mora]}."""
        nombres = list(self.get_feature_names_out())
        grupos = {c: [nombres.index(c)] for c in self.numericas}
        for ind in self.indicadoras:
            grupos.setdefault(INDICADOR_DE.get(ind, ind), []).append(nombres.index(ind))
        for col in self.categoricas:
            grupos[col] = [i for i, n in enumerate(nombres) if n.startswith(f"{col}_")]
        return grupos
