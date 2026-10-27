"""Velocímetro del puntaje en SVG propio (se incrusta en el HTML de la tarjeta)."""

import math

from formato import numero
from motor import BANDAS

ANCHO, ALTO = 340, 200
CX, CY, R, GROSOR = 170, 172, 142, 24


def _punto(t, radio):
    """Punto del semicírculo para t en [0, 1] (0 = extremo izquierdo, 1 = derecho)."""
    angulo = math.pi * (1 - t)
    return CX + radio * math.cos(angulo), CY - radio * math.sin(angulo)


def _arco(t0, t1):
    """Sector del anillo entre t0 y t1 (cada banda ocupa menos de media vuelta)."""
    (x0, y0), (x1, y1) = _punto(t0, R), _punto(t1, R)
    (x2, y2), (x3, y3) = _punto(t1, R - GROSOR), _punto(t0, R - GROSOR)
    r2 = R - GROSOR
    return (f"M{x0:.2f},{y0:.2f} A{R},{R} 0 0 1 {x1:.2f},{y1:.2f} "
            f"L{x2:.2f},{y2:.2f} A{r2},{r2} 0 0 0 {x3:.2f},{y3:.2f} Z")


def velocimetro(puntaje, percentil, cortes, tinta, tinta_suave):
    """SVG con las cinco bandas de riesgo, un marcador y el puntaje al centro.

    El arco representa la posición frente a la población: cada banda es un 20 % de los
    solicitantes (de izquierda a derecha, de más a menos riesgo) y el marcador va en el
    percentil de la persona. cortes = {banda: puntaje donde empieza} (Modelo.cortes_bandas)
    se usa solo para el texto al pasar el mouse."""
    orden = list(reversed(BANDAS))                        # de "Riesgo alto" a "Excelente"
    separacion = 0.005
    partes = []
    for i, (inicio, nombre, color, _) in enumerate(orden):
        t0, t1 = i / len(orden) + separacion, (i + 1) / len(orden) - separacion
        desde = f"desde {numero(cortes[nombre])} puntos" if nombre in cortes else f"menos de {numero(cortes[orden[1][1]])} puntos"
        partes.append(f'<path d="{_arco(t0, t1)}" fill="{color}"><title>{nombre}: percentil {inicio} a '
                      f'{inicio + 20} ({desde})</title></path>')

    # marcador sobre el arco, en el percentil de la persona (sin aguja: no tapa el número)
    xm, ym = _punto(min(max(percentil / 100, 0.0), 1.0), R - GROSOR / 2)
    partes.append(f'<circle cx="{xm:.2f}" cy="{ym:.2f}" r="15" fill="#ffffff" stroke="{tinta}" '
                  f'stroke-width="5"><title>Tu puntaje: {numero(puntaje)} (percentil {numero(percentil)})</title></circle>')
    (xi, yi), (xd, yd) = _punto(0, R - GROSOR / 2), _punto(1, R - GROSOR / 2)
    partes.append(f'<text x="{xi:.1f}" y="{yi + 22:.1f}" text-anchor="middle" font-size="12" '
                  f'fill="{tinta_suave}">Más riesgo</text>'
                  f'<text x="{xd:.1f}" y="{yd + 22:.1f}" text-anchor="middle" font-size="12" '
                  f'fill="{tinta_suave}">Menos riesgo</text>')
    partes.append(f'<text x="{CX}" y="{CY - 26}" text-anchor="middle" font-size="58" font-weight="800" '
                  f'letter-spacing="-2" fill="{tinta}">{numero(puntaje)}</text>'
                  f'<text x="{CX}" y="{CY - 2}" text-anchor="middle" font-size="13" font-weight="600" '
                  f'letter-spacing="1.5" fill="{tinta_suave}">PUNTOS</text>')
    return (f'<svg viewBox="0 0 {ANCHO} {ALTO + 12}" width="100%" style="max-width:{ANCHO}px; display:block; '
            f'margin:0 auto; font-family:Inter, sans-serif" role="img" '
            f'aria-label="Puntaje {numero(puntaje)}, percentil {numero(percentil)} de los solicitantes">'
            + "".join(partes) + "</svg>")
