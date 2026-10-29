"""La figura de la serie 5–200 mV/s, hecha con el MOTOR DE LA VENTANA.

    uv run python -I analisis_CV2/figuras.py <archivo .mpr> ...

No dibuja nada por su cuenta. Monta la misma `EchemSession` que usa la
sección de electroquímica, elige paneles del mismo catálogo y llama a
`save_figure`, que es el botón «Guardar figura…» de la pestaña Figura.

**Eso es el punto.** La versión anterior de este archivo eran ciento
sesenta líneas de matplotlib escritas a mano, y una figura escrita a
mano se separa de lo que enseña el programa en cuanto uno de los dos
cambia — además de dejar fuera a quien no escriba matplotlib. Lo que
queda aquí es la elección: qué paneles, en qué orden, a qué ancho.
"""
import sys
from pathlib import Path

from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.theme import PALETTES

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_CV2")
MASS_MG = 0.020
ELECTROLYTE = "KOH_3M"

#: Los cuatro paneles del informe: los voltamperogramas, lo que dan, lo
#: que se fuga y lo que el estudio de velocidades resuelve.
PANELS = ("cv_serie", "cv_capacitancia", "cv_carga", "cv_terminos")


def main(paths):
    session = EchemSession(name="S1 900 °C")
    session.electrode.mass_mg = MASS_MG
    session.electrode.use_electrolyte(ELECTROLYTE)
    for path in paths:
        if not session.load_cv(path, into_series=True):
            raise SystemExit(session.messages[-1][1])
    session.rate_series.sort(key=lambda curve: curve.scan_rate)
    session.cv = session.rate_series[-1]
    session.retag_electrode()
    if session.analyse() is None:
        raise SystemExit("el análisis falló")

    session.set_figure_panels(PANELS)
    session.use_figure_preset("articulo_2col")
    for suffix in ("png", "pdf"):
        if not session.save_figure(OUT / f"cv_serie.{suffix}",
                                   PALETTES["claro"]):
            raise SystemExit("no se pudo guardar la figura")
    for level, text in session.messages:
        if level == "error":
            print("✗", text)
    print("escrita cv_serie.{png,pdf}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
