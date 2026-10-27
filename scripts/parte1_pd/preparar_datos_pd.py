"""
Genera los datos de la Parte 1 (probabilidad de incumplimiento) a partir del archivo original.

    python -m scripts.parte1_pd.preparar_datos_pd                          # lee datos/archive.zip o datos/loan/loan.csv
    python -m scripts.parte1_pd.preparar_datos_pd --origen ruta/loan.csv

Resultado: data/processed/pd_limpio.parquet, con una fila por crédito y las columnas
    y            1 = malo, 0 = bueno (codificación del enunciado a partir de loan_status)
    particion    train / valid / test
    issue_d      mes de originación
    + las variables de entrada ya limpias (sin imputar ni escalar: eso se ajusta con train
      en src/preprocesamiento.py y se guarda junto al modelo).

Las reglas de limpieza son las de María Fernanda Valencia (Punto_1/Primer_Punto.ipynb y
Punto_1/Ideas_primer_punto.docx). Aquí solo van las reglas fijas, que no aprenden nada de los
datos y por eso se pueden aplicar igual a todos los conjuntos.

Población y partición (ver la justificación en el reporte):
    - Solo créditos a 36 meses. Los datos se cortan en dic-2015 y los créditos a 60 meses
      originados desde 2011 no habían terminado: solo vemos los que acabaron temprano, que son
      sobre todo los malos (censura). Ningún crédito a 60 meses posterior a 2010 se puede evaluar.
    - Partición temporal por fecha de originación:
          train  jun-2007 a dic-2011
          valid  ene-2012 a jun-2012
          test   jul-2012 a dic-2012
      2013-2015 se excluye: los créditos a 36 meses de esos años vencen después de dic-2015.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ORIGENES = [Path("datos/loan/loan.csv"), Path("Punto_1/loan.csv"), Path("datos/archive.zip")]
SALIDA = Path("data/processed/pd_limpio.parquet")

# Codificación de loan_status pedida en el enunciado. Las categorías que no están aquí
# (Current, Issued, In Grace Period, Late (16-30 days)) quedan como NA y se excluyen.
CODIFICACION = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}
ESTADOS_NA = {"Current", "Issued", "In Grace Period", "Late (16-30 days)"}

# Solo información disponible al momento de la solicitud (nada posterior al desembolso)
COLUMNAS = ["loan_status", "issue_d", "loan_amnt", "term", "emp_length", "home_ownership",
            "annual_inc", "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
            "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]

# Sin estas variables el solicitante no tiene historial crediticio (29 casos, todos de 2007)
COLS_HISTORIAL = ["delinq_2yrs", "inq_last_6mths", "pub_rec", "open_acc", "total_acc",
                  "earliest_cr_line"]

MESES = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
         "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def indice_mes(s):
    """'Dec-2011' -> 2011*12 + 12. La resta de dos índices = meses transcurridos.
    No depende del idioma del sistema."""
    s = s.astype(str).str.strip()
    return pd.to_numeric(s.str[-4:], errors="coerce") * 12 + s.str[:3].map(MESES)


FIN_TRAIN = indice_mes(pd.Series(["Dec-2011"]))[0]
FIN_VALID = indice_mes(pd.Series(["Jun-2012"]))[0]
FIN_TEST = indice_mes(pd.Series(["Dec-2012"]))[0]


def leer_original(origen=None):
    if origen is None:
        origen = next((r for r in ORIGENES if r.exists()), None)
        if origen is None:
            raise FileNotFoundError("No se encontró loan.csv ni datos/archive.zip (ver README).")
    origen = Path(origen)
    print(f"Leyendo {origen}")
    if origen.suffix == ".zip":
        with zipfile.ZipFile(origen) as z:
            nombre = next(n for n in z.namelist() if n.endswith("loan.csv"))
            with z.open(nombre) as f:
                return pd.read_csv(f, usecols=COLUMNAS, low_memory=False)
    return pd.read_csv(origen, usecols=COLUMNAS, low_memory=False)


def codificar_objetivo(d):
    estado = d["loan_status"].str.strip()
    desconocidos = set(estado.unique()) - set(CODIFICACION) - ESTADOS_NA
    assert not desconocidos, f"Categorías de loan_status no contempladas: {desconocidos}"
    d = d.assign(y=estado.map(CODIFICACION))
    return d.dropna(subset=["y"]).astype({"y": int}).drop(columns="loan_status")


def asignar_particion(d):
    mes = indice_mes(d["issue_d"])
    particion = np.select([mes <= FIN_TRAIN, mes <= FIN_VALID, mes <= FIN_TEST],
                          ["train", "valid", "test"], default="fuera_censurado")
    return d.assign(particion=particion)


def limpieza_fija(d):
    """Reglas sin parámetros aprendidos (María Fernanda Valencia)."""
    d = d.dropna(subset=COLS_HISTORIAL).copy()

    # Un 0 en mths_since_last_delinq con delinq_2yrs == 0 es contradictorio ("mora hace 0
    # meses" sin ninguna mora en 2 años): era el código de "sin mora" en 2007-2008.
    codigo = (d["mths_since_last_delinq"] == 0) & (d["delinq_2yrs"] == 0)
    d.loc[codigo, "mths_since_last_delinq"] = np.nan
    d["nunca_mora"] = d["mths_since_last_delinq"].isna().astype(int)

    # Antigüedad laboral: "< 1 year" -> 0, "10+ years" -> 10, "n/a" -> NaN (+ indicador)
    texto = d["emp_length"].astype(str)
    anios = pd.to_numeric(texto.str.extract(r"(\d+)")[0], errors="coerce")
    anios[texto.str.startswith("<")] = 0
    d["emp_length"] = anios.astype(float)
    d["emp_length_na"] = d["emp_length"].isna().astype(int)

    # Antigüedad crediticia en meses respecto a la originación (no a hoy: eso sería futuro)
    d["antig_credito_meses"] = indice_mes(d["issue_d"]) - indice_mes(d["earliest_cr_line"])
    assert (d["antig_credito_meses"] >= 0).all(), "Antigüedad crediticia negativa"

    # El diccionario solo documenta RENT, OWN, MORTGAGE y OTHER
    d["home_ownership"] = (d["home_ownership"].astype(str).str.upper()
                           .replace({"NONE": "OTHER", "ANY": "OTHER"}))

    # Sin saldo rotativo no hay utilización que calcular: es 0, no un dato perdido
    sin_rotativo = d["revol_util"].isna() & (d["revol_bal"] == 0)
    d.loc[sin_rotativo, "revol_util"] = 0.0

    # Registros públicos: casi toda la masa está en 0 y 1 -> categorías 0, 1, 2 o más
    d["pub_rec"] = d["pub_rec"].clip(upper=2)

    return d.drop(columns=["earliest_cr_line", "term"])


def preparar(datos):
    d = codificar_objetivo(datos)
    d = d[d["term"].astype(str).str.strip() == "36 months"]
    d = asignar_particion(d)
    d = d[d["particion"] != "fuera_censurado"]
    return limpieza_fija(d).reset_index(drop=True)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", default=None, help="loan.csv o el zip de Kaggle")
    parser.add_argument("--salida", default=str(SALIDA))
    args = parser.parse_args(argumentos)

    datos = leer_original(args.origen)
    print(f"Archivo original: {datos.shape[0]} filas")
    d = preparar(datos)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(salida, index=False, compression="zstd")

    resumen = (d.groupby("particion")["y"].agg(n="size", malos="sum", tasa_malos="mean")
                .reindex(["train", "valid", "test"]))
    print(resumen.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nGuardado {salida} ({d.shape[0]} filas, {d.shape[1]} columnas)")


if __name__ == "__main__":
    main()
