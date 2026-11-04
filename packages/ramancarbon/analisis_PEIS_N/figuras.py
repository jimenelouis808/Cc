"""La figura de los cuatro PEIS, hecha con el MOTOR DE LA VENTANA.

    uv run python -I analisis_PEIS_N/figuras.py E.mpr EA.mpr EB.mpr EP.mpr

No dibuja nada por su cuenta: monta la misma `EchemSession` que usa la
sección de electroquímica, elige paneles del mismo catálogo y llama a
`save_figure`, que es el botón «Guardar figura…» de la pestaña Figura.
Lo que queda aquí es la ELECCIÓN —qué paneles, en qué orden, a qué
ancho— y no doscientas líneas de matplotlib que se separan de la ventana
en cuanto una de las dos cambia.
"""
import sys
from pathlib import Path

from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.theme import PALETTES

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS_N")
LABELS = ("E (etanol)", "EA (+acetonitrilo)", "EB (+bencilamina)",
          "EP (+piridina)")

#: Seis paneles, en el orden en que se leen. El rincón va aparte del
#: Nyquist porque el Nyquist entero se dibuja a escala uno a uno —la
#: forma ES el contenido del diagrama— y a esa escala el tramo donde
#: vive el poro mide pixeles.
PANELS = ("eis_nyquist", "eis_rincon", "eis_bode", "eis_capacitancia",
          "eis_drt", "eis_residuo")


def main(paths):
    session = EchemSession(name="carbones Fe/Fe3C por precursor de N")
    session.electrode.use_electrolyte("H2SO4_0.5M")
    session.electrode.reference = "Ag/AgCl_sat"
    session.electrode.use_disc(3.0)
    # UN circuito para las cuatro, y el que ganó el barrido del
    # catálogo entero: dos arcos. El automático elegía uno distinto por
    # espectro —R0-(R1|Q1) en dos de ellos— y con ése la curva ajustada
    # no seguía a los puntos: 17.8 % de residuo medio en EP. Con dos
    # arcos baja a 5.1 % ahí y a 0.5-0.7 % en E y EA, y además las
    # cuatro quedan comparables, que es lo que un circuito común es
    # para.
    session.circuit = "R0-(R1|Q1)-(R2|Q2)"
    for path, label in zip(paths, LABELS):
        if not session.add_eis_to_series(path, label=label):
            raise SystemExit(session.messages[-1][1])

    session.set_figure_panels(PANELS)
    session.use_figure_preset("articulo_2col")
    session.composite.panel_height_in = 3.1
    for suffix in ("png", "pdf"):
        if not session.save_figure(OUT / f"peis_precursores.{suffix}",
                                   PALETTES["claro"]):
            raise SystemExit("no se pudo guardar la figura")
    for level, text in session.messages:
        if level == "error":
            print("✗", text)
    print("escrita peis_precursores.{png,pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
