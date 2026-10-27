"""Números con el formato colombiano: coma decimal, punto de miles y espacio antes de «%»."""


def numero(x, decimales=0, signo=False):
    """1234.5 -> '1.234,5'; con signo=True: '+1.234,5'."""
    texto = f"{x:{'+' if signo else ''},.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def porcentaje(fraccion, decimales=1, signo=False):
    """0.102 -> '10,2 %'."""
    return f"{numero(100 * fraccion, decimales, signo)} %"


def puntos_porcentuales(diferencia, decimales=1):
    """Diferencia entre dos fracciones en puntos porcentuales: 0.011 -> '+1,1 p. p.'."""
    return f"{numero(100 * diferencia, decimales, signo=True)} p. p."


def usd(x):
    return f"USD {numero(x)}"
