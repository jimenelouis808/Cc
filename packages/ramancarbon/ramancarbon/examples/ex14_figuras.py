"""El motor de figuras: la misma gráfica, lista para tres destinos.

    python -m ramancarbon.examples.ex14_figuras

Una figura aquí es un *valor*: series, estilo, marcas, bandas, notas y
recuadro en un objeto que se guarda, se reabre y se vuelve a dibujar. Eso
permite lo que hace este ejemplo: dibujar una vez y exportar con el ancho
de columna de una revista, con los tamaños de una presentación y en
grises, sin tocar los datos.
"""

from __future__ import annotations

from pathlib import Path


from ramancarbon.examples.demo_data import make_demo, make_xrd_demo
from ramancarbon.plotting import (
    AxisStyle,
    Inset,
    Plot,
    SecondaryAxis,
    Series,
    from_pattern,
    from_spectrum,
    preset,
)
from ramancarbon.plotting.layout import grid

SALIDA = Path("figuras_demo")


def cascada() -> Plot:
    """Seis espectros apilados, normalizados y con eje de nm arriba."""
    nombres = ["SWCNT", "DWCNT", "MWCNT", "grafeno_1L", "GO", "MWCNT_FeSe"]
    plot = Plot(name="cascada")
    for indice, nombre in enumerate(nombres):
        plot.add(from_spectrum(make_demo(nombre, seed=indice), label=nombre))
    plot.style = preset("cascada").replace(
        x=AxisStyle(label="Desplazamiento Raman (cm⁻¹)", limits=(200.0, 3000.0)),
        y=AxisStyle(label="Intensidad normalizada (u.a.)", show_minor=False),
        secondary_x=SecondaryAxis(kind="lambda", parameter=532.0),
    )
    plot.shade(1300.0, 1400.0, label="banda D")
    plot.shade(1560.0, 1620.0, label="banda G")
    return plot


def difractograma() -> Plot:
    """Un patrón con eje de d arriba, escala de raíz y recuadro."""
    patron = make_xrd_demo("CNT_FeSe")
    plot = Plot(name="drx", series=[from_pattern(patron, label="CNT@FeSe")])
    plot.style = preset("acs").replace(
        x=AxisStyle(label="2θ (°)", limits=(10.0, 80.0), minor_ticks=5),
        y=AxisStyle(label="Intensidad (cuentas)", scale="sqrt"),
        secondary_x=SecondaryAxis(kind="d", parameter=patron.wavelength),
        annotate_peaks=False,
    )
    plot.mark(26.5, 28.5, 44.6, label="", colour="#999999")
    plot.inset = Inset(x_limits=(24.0, 32.0), position=(0.55, 0.5, 0.4, 0.42))
    return plot


def voltamperograma() -> Plot:
    """Cuatro velocidades, con símbolos que sobreviven a la impresión."""
    from ramancarbon.examples.demo_data import cv_rate_series

    plot = Plot(name="cv")
    for indice, cv in enumerate(cv_rate_series("pseudocondensador", (0.005, 0.02, 0.05, 0.1))):
        plot.add(Series(
            x=cv.potential, y=cv.current * 1e3,
            label=f"{cv.scan_rate * 1e3:.0f} mV s⁻¹",
            marker_every=40,
        ))
    plot.style = preset("acs").replace(
        x=AxisStyle(label="E vs. Ag/AgCl (V)"),
        y=AxisStyle(label="Corriente (mA)"),
        cycle_markers=True, marker="o", marker_size=3.0,
    )
    plot.mark(0.0, axis="y", colour="#666666")
    return plot


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    figura = cascada()

    print("La misma figura, tres destinos. Cambia el estilo, no los datos.\n")
    for destino in ("acs", "presentacion", "grises"):
        estilo = preset(destino).replace(
            x=figura.style.x, y=figura.style.y,
            secondary_x=figura.style.secondary_x,
            offset=figura.style.offset, normalise=figura.style.normalise,
        )
        copia = Plot(series=figura.series, style=estilo,
                     bands=figura.bands, name=f"cascada_{destino}")
        ruta = copia.save(SALIDA / f"cascada_{destino}.png")
        print(f"  {destino:14s} {estilo.width_in:4.2f} × {estilo.height_in:4.2f} in, "
              f"{estilo.dpi} ppp, texto {estilo.font_size:.0f} pt → {ruta.name}")

    print("\nY los números que se dibujaron, no los de origen:")
    datos = figura.save_data(SALIDA / "cascada.csv")
    columnas, filas = figura.data_table()
    print(f"  {datos.name}: {len(columnas)} columnas × {len(filas)} filas")
    print(f"  columnas: {', '.join(columnas[:4])}, …")
    print("  Llevan aplicados el desplazamiento y la normalización, que es")
    print("  la única forma de que quien lea el archivo vea la misma curva.\n")

    paneles = grid([cascada(), difractograma(), voltamperograma()], columns=2)
    paneles.name = "resumen"
    paneles.width_in = 7.0
    paneles.height_in = 5.6
    ruta = paneles.save(SALIDA / "panel.png")
    archivos = paneles.save_data(SALIDA / "panel_datos")
    print(f"Figura de paneles: {ruta.name}, etiquetas (a)–(c),")
    print(f"  un archivo de datos por panel: {', '.join(f.name for f in archivos)}")

    print("\nY la figura entera cabe en un diccionario:")
    payload = figura.to_dict(include_data=False)
    print(f"  {len(payload['series'])} series, {len(payload['style'])} campos de estilo,")
    print("  sin los datos — el proyecto apunta al archivo de origen.")
    print(f"\nTodo en {SALIDA.resolve()}")


if __name__ == "__main__":
    main()
