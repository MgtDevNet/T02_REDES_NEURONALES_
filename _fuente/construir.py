"""
Genera la entrada de blog a partir del informe en PDF, sin tocar su contenido.

    python construir.py informe.pdf salida/

Cada página del cuerpo se conserva como gráfico vectorial; solo se recortan el encabezado
y el número de página. Encima van una capa de texto invisible (búsqueda y copia) y los
enlaces del PDF. La portada y los índices se reemplazan por los de la página web.
"""
import html
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import pymupdf

PRIMERA = 6                     # primera página del cuerpo; antes van portada e índices
SUP, INF = 55, 785              # franja entre el encabezado y el número de página
ACENTOS = {"´": "́", "˜": "̃", "¨": "̈"}


def limpiar(texto):
    """Une los acentos que TeX guarda aparte (´o -> ó) para que la búsqueda funcione."""
    texto = re.sub(r"([´˜¨])\s?([A-Za-zıȷ])",
                   lambda m: m[2].replace("ı", "i").replace("ȷ", "j") + ACENTOS[m[1]], texto)
    return unicodedata.normalize("NFC", texto)


def contenido(pagina):
    rects = [pymupdf.Rect(l["bbox"]) for b in pagina.get_text("dict")["blocks"] if b["type"] == 0
             for l in b["lines"] if "".join(s["text"] for s in l["spans"]).strip()]
    rects += [d["rect"] for d in pagina.get_drawings()]
    rects += [pymupdf.Rect(i["bbox"]) for i in pagina.get_image_info()]
    fuera = [r for r in rects if r.y1 <= SUP or r.y0 >= INF]
    dentro = [r for r in rects if r.y1 > SUP and r.y0 < INF]
    for r in fuera:
        texto = pagina.get_textbox(r).strip()
        assert r.height < 1.5 or texto.isdigit() or texto.startswith("Redes Neuronales"), (pagina.number, texto)
    return dentro


def recortes(doc):
    cajas = {}
    for n in range(PRIMERA - 1, len(doc)):
        r = contenido(doc[n])
        cajas[n] = (min(x.x0 for x in r), min(x.y0 for x in r), max(x.x1 for x in r), max(x.y1 for x in r))
        assert all(x.y1 <= INF + 1 and x.y0 >= SUP - 1 for x in r), n
    x0 = min(c[0] for c in cajas.values()) - 8
    x1 = max(c[2] for c in cajas.values()) + 8
    return {n: (x0, c[1] - 5, x1, c[3] + 5) for n, c in cajas.items()}


def svg_recortado(pagina, caja):
    x0, y0, x1, y1 = caja
    svg = pagina.get_svg_image(text_as_path=True)
    caja_vb = f'viewBox="{x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f}"'
    svg, n = re.subn(r'width="[^"]+" height="[^"]+" viewBox="[^"]+"',
                     f'width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" {caja_vb}', svg, count=1)
    assert n == 1
    return svg


def posicion(doc, destino, cajas):
    """Devuelve (página, % vertical dentro del recorte) de un destino interno."""
    n = destino["page"]
    if n not in cajas:
        return None
    alto = doc[n].rect.height
    y = alto - destino["to"].y          # el PDF mide desde abajo
    x0, y0, x1, y1 = cajas[n]
    return n, max(0, min(100, (y - y0) / (y1 - y0) * 100))


def capa_texto(pagina, caja):
    x0, y0, x1, y1 = caja
    ancho, alto = x1 - x0, y1 - y0
    partes = []
    for b in pagina.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            texto = limpiar("".join(s["text"] for s in l["spans"])).strip()
            r = pymupdf.Rect(l["bbox"])
            if not texto or r.y1 <= SUP or r.y0 >= INF:
                continue
            partes.append(
                f'<span style="left:{(r.x0 - x0) / ancho * 100:.2f}%;top:{(r.y0 - y0) / alto * 100:.2f}%;'
                f'width:{r.width / ancho * 100:.2f}%;font-size:{r.height / ancho * 100:.2f}cqw">'
                f"{html.escape(texto)}</span>")
    return "".join(partes)


def enlaces(doc, pagina, caja, cajas):
    x0, y0, x1, y1 = caja
    ancho, alto = x1 - x0, y1 - y0
    partes = []
    for l in pagina.get_links():
        r = l["from"]
        if r.y1 <= SUP or r.y0 >= INF:
            continue
        if l["kind"] == pymupdf.LINK_URI:
            href, extra = html.escape(l["uri"]), ' target="_blank" rel="noopener"'
        else:
            p = posicion(doc, l, cajas)
            if p is None:
                continue
            href, extra = f"#d{l['xref']}", ""
        partes.append(f'<a href="{href}"{extra} style="left:{(r.x0 - x0) / ancho * 100:.2f}%;'
                      f'top:{(r.y0 - y0) / alto * 100:.2f}%;width:{r.width / ancho * 100:.2f}%;'
                      f'height:{r.height / alto * 100:.2f}%" aria-label="Enlace"></a>')
    return "".join(partes)


def indice(doc, cajas):
    """Árbol del índice a partir de los marcadores del PDF."""
    items, anclas = [], {}
    for i, (nivel, titulo, _, dest) in enumerate(doc.get_toc(simple=False)):
        titulo = titulo.strip()
        p = posicion(doc, dest, cajas)
        if not titulo or p is None:
            continue
        anclas.setdefault(p[0], []).append((f"s{i}", p[1], titulo, nivel))
        items.append((nivel, titulo, f"s{i}"))
    return items, anclas


def html_indice(items):
    raiz = []
    pila = [(0, raiz)]
    for nivel, titulo, ancla in items:
        hijos = []
        while pila[-1][0] >= nivel:
            pila.pop()
        pila[-1][1].append((nivel, titulo, ancla, hijos))
        pila.append((nivel, hijos))

    def lista(nodos):
        return "<ul>" + "".join(
            f'<li class="n{n}"><a href="#{a}">{html.escape(t)}</a>{lista(h) if h else ""}</li>'
            for n, t, a, h in nodos) + "</ul>"
    return lista(raiz)


def listas(doc, cajas):
    """Índices de figuras y cuadros: enlaces de las páginas 4 y 5 del PDF."""
    filas = {}
    for n in (3, 4):
        for l in doc[n].get_links():
            nombre = l.get("nameddest", "")
            if not nombre.startswith(("figure.", "table.")):
                continue
            texto = limpiar(" ".join(doc[n].get_textbox(l["from"]).split()))
            p = posicion(doc, l, cajas)
            if nombre in filas:
                previo = filas[nombre][0]
                filas[nombre][0] = previo[:-1] + texto if previo.endswith("-") else f"{previo} {texto}"
            elif p and texto:
                filas[nombre] = [texto, l["xref"], p, nombre]
    figuras = [tuple(f[:3]) for f in filas.values() if f[3].startswith("figure.")]
    cuadros = [tuple(f[:3]) for f in filas.values() if f[3].startswith("table.")]
    return figuras, cuadros


def main(pdf, salida):
    salida = Path(salida)
    (salida / "paginas").mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    cajas = recortes(doc)
    items, anclas = indice(doc, cajas)
    figuras, cuadros = listas(doc, cajas)

    # destinos de todos los enlaces internos, agrupados por página
    destinos = {}
    for n in cajas:
        for l in doc[n].get_links():
            if l["kind"] != pymupdf.LINK_URI:
                p = posicion(doc, l, cajas)
                if p:
                    destinos.setdefault(p[0], {})[f"d{l['xref']}"] = p[1]
    for _, xref, p in figuras + cuadros:
        destinos.setdefault(p[0], {})[f"d{xref}"] = p[1]

    bloques = []
    for n, caja in cajas.items():
        pagina = doc[n]
        (salida / "paginas" / f"p{n + 1:02d}.svg").write_text(svg_recortado(pagina, caja), encoding="utf-8")
        ancho, alto = caja[2] - caja[0], caja[3] - caja[1]
        marcas = "".join(f'<span class="ancla" id="{k}" style="top:{v:.2f}%"></span>'
                         for k, v in destinos.get(n, {}).items())
        marcas += "".join(f'<h{min(nivel + 1, 4)} class="ancla titulo" id="{k}" style="top:{v:.2f}%">{html.escape(t)}</h{min(nivel + 1, 4)}>'
                          for k, v, t, nivel in anclas.get(n, []))
        bloques.append(
            f'<div class="hoja" style="aspect-ratio:{ancho:.2f}/{alto:.2f}" data-pagina="{n + 1}">'
            f'<img src="paginas/p{n + 1:02d}.svg" alt="" width="{ancho:.0f}" height="{alto:.0f}" '
            f'loading="{"eager" if n < PRIMERA + 2 else "lazy"}" decoding="async">'
            f'{marcas}<div class="texto">{capa_texto(pagina, caja)}</div>{enlaces(doc, pagina, caja, cajas)}</div>')

    # escudo de la portada original
    marco = doc[0].get_image_rects(doc[0].get_images()[0][0])[0]
    doc[0].get_pixmap(clip=marco, dpi=220, alpha=True).save(salida / "escudo.png")

    plantilla = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
    lista = lambda filas: "".join(f'<li><a href="#d{x}">{html.escape(t)}</a></li>' for t, x, _ in filas)
    pagina_final = (plantilla
                    .replace("{{ESCUDO}}", "escudo.png")
                    .replace("{{INDICE}}", html_indice(items))
                    .replace("{{FIGURAS}}", lista(figuras))
                    .replace("{{CUADROS}}", lista(cuadros))
                    .replace("{{CUERPO}}", "".join(bloques)))
    (salida / "index.html").write_text(pagina_final, encoding="utf-8")
    shutil.copy(pdf, salida / "informe.pdf")
    (salida / ".nojekyll").touch()
    print(len(cajas), "páginas;", len(items), "títulos;", len(figuras), "figuras;", len(cuadros), "cuadros")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
