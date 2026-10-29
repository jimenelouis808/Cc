"""La figura de los tres PEIS, hecha con el MOTOR DE LA VENTANA.

    uv run python -I analisis_PEIS/figuras.py <800>.mpr <900>.mpr <1020>.mpr

No dibuja nada por su cuenta. Monta la misma `EchemSession` que usa la
sección de electroquímica, elige paneles del mismo catálogo y llama a
`save_figure`, que es el botón «Guardar figura…» de la pestaña Figura.

**Eso es el punto.** La versión anterior de este archivo era doscientas
líneas de matplotlib escritas a mano, y una figura escrita a mano se
separa de lo que enseña el programa en cuanto uno de los dos cambia —
además de dejar fuera a quien no escriba matplotlib. Lo que queda aquí
es la elección: qué paneles, en qué orden, a qué ancho.
"""
import sys
from pathlib import Path

from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.theme import PALETTES

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS")
LABELS = ("800 °C", "900 °C", "1020 °C")

#: Los seis paneles del informe, en el orden en que se leen: el espectro
#: entero, el rincón donde está el poro, el Bode con el codo, la
#: capacitancia sin circuito, la DRT y el residuo.
PANELS = ("eis_nyquist", "eis_rincon", "eis_bode", "eis_capacitancia",
          "eis_drt", "eis_residuo")


def main(paths):
    session = EchemSession(name="carbón en H2SO4 0.5 M")
    session.electrode.use_electrolyte("H2SO4_0.5M")
    session.electrode.use_disc(3.0)
    for path, label in zip(paths, LABELS):
        if not session.add_eis_to_series(path, label=label):
            raise SystemExit(session.messages[-1][1])

    session.set_figure_panels(PANELS)
    session.use_figure_preset("articulo_2col")
    session.composite.panel_height_in = 3.1
    for suffix in ("png", "pdf"):
        if not session.save_figure(OUT / f"peis_serie.{suffix}",
                                   PALETTES["claro"]):
            raise SystemExit("no se pudo guardar la figura")
    for level, text in session.messages:
        if level == "error":
            print("✗", text)
    print("escrita peis_serie.{png,pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
