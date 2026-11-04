"""Las figuras de los cuatro PEIS, hechas con el MOTOR DE LA VENTANA.

    uv run python -I analisis_PEIS_N/figuras.py E.mpr EA.mpr EB.mpr EP.mpr

No dibuja nada por su cuenta: monta la misma `EchemSession` que usa la
sección de electroquímica, elige paneles del mismo catálogo y llama a
`save_figure`, que es el botón «Guardar figura…» de la pestaña Figura.
Lo que queda aquí es la ELECCIÓN —qué paneles, en qué orden, a qué
ancho— y no doscientas líneas de matplotlib que se separan de la ventana
en cuanto una de las dos cambia.

Salen DOS archivos:

- `peis_precursores` — los seis paneles de siempre. El Nyquist va a
  escala uno a uno y el rincón de alta frecuencia va en su propio panel,
  así que el recuadro dentro del Nyquist **no se dibuja**: la misma
  década dos veces en una figura gasta un cuarto del panel en algo que
  el lector tiene al lado, y en un Nyquist ese cuarto es justo donde va
  la rama de baja frecuencia.
- `peis_nyquist_lineal_vs_log` — el mismo Nyquist en las dos escalas,
  lado a lado. Es la comparación dibujada: a la izquierda se pueden leer
  ángulos y diámetros y no se ve el rincón; a la derecha se ven las
  cuatro décadas y NO se puede leer ni un ángulo de depresión ni un
  diámetro de arco. Lo único que sobrevive al logaritmo es la PENDIENTE.
"""
import sys
from pathlib import Path

import numpy as np

from ramancarbon.echem.biologic import read_mpr
from ramancarbon.gui.echem_state import EchemSession
from ramancarbon.gui.theme import PALETTES

OUT = Path("/home/user/Cc/packages/ramancarbon/analisis_PEIS_N")

#: El nombre corto de cada muestra, en el orden en que se pasan. El
#: potencial NO va aquí: se lee del propio archivo y se le pega detrás,
#: para que una etiqueta no pueda decir un número que el archivo no
#: tiene.
NAMES = ("E", "EA", "EB", "EP")

#: Seis paneles, en el orden en que se leen.
PANELS = ("eis_nyquist", "eis_rincon", "eis_bode", "eis_capacitancia",
          "eis_drt", "eis_residuo")

#: Y la figura de la comparación de escalas, que es su propia discusión.
SCALES_PANELS = ("eis_nyquist", "eis_nyquist_log")

#: UN circuito para las cuatro, y el que ganó el barrido del catálogo
#: entero. El automático elegía uno distinto por espectro —R0-(R1|Q1) en
#: dos de ellos— y con ése la curva ajustada no seguía a los puntos:
#: 17.8 % de residuo medio en EP. Además, parámetros de circuitos
#: distintos no son los mismos parámetros, que es lo que un circuito
#: común es para.
CIRCUIT = "R0-(R1|Q1)-(R2|Q2)"


def label_for(name: str, path: str) -> str:
    """``E (+0.45 V)``: el nombre y el continuo al que se midió.

    El potencial sale del archivo, no de una tabla escrita a mano. En E,
    EA y EP es el OCP de cada muestra —sus nombres dicen «PEIS 0 V», que
    en EC-Lab es 0 V *frente al circuito abierto*— y en EB es un
    potencial APLICADO. La etiqueta no distingue las dos cosas porque no
    cabe; las tablas sí.
    """
    column = read_mpr(path).columns.get("<Ewe>/V")
    if column is None:
        return name
    return f"{name} ({float(np.mean(column)):+.2f} V)"


def build(paths):
    session = EchemSession(name="carbones Fe/Fe3C por precursor de N")
    session.electrode.use_electrolyte("H2SO4_0.5M")
    session.electrode.reference = "Ag/AgCl_sat"
    session.electrode.use_disc(3.0)
    session.electrode.mass_mg = 0.909
    session.circuit = CIRCUIT
    for path, name in zip(paths, NAMES):
        if not session.add_eis_to_series(path, label=label_for(name, path)):
            raise SystemExit(session.messages[-1][1])
    return session


def save(session, panels, stem, height_in):
    session.set_figure_panels(panels)
    session.use_figure_preset("articulo_2col")
    session.composite.panel_height_in = height_in
    for suffix in ("png", "pdf"):
        if not session.save_figure(OUT / f"{stem}.{suffix}",
                                   PALETTES["claro"]):
            raise SystemExit(f"no se pudo guardar {stem}")
    print(f"escrita {stem}.{{png,pdf}}")


def main(paths):
    session = build(paths)
    save(session, PANELS, "peis_precursores", 3.1)
    # El Nyquist cuadra, así que su panel va más alto que ancho: a dos
    # columnas y dos paneles, cada caja mide 3.6 pulgadas de ancho.
    save(session, SCALES_PANELS, "peis_nyquist_lineal_vs_log", 3.7)
    for level, text in session.messages:
        if level == "error":
            print("✗", text)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
