"""Importar y exportar: una carpeta revuelta, ordenada sola.

    python -m ramancarbon.examples.ex15_datos

Cuatro instrumentos producen archivos que se llaman todos igual. Este
ejemplo escribe una carpeta con las cinco medidas y un archivo que no es
una medida, y deja que el programa averigüe qué es cada cosa mirando los
números, no la extensión.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ramancarbon.core.io import write_spectrum
from ramancarbon.dataio import Project, detect, load_folder, scan
from ramancarbon.dataio.export import summary_table
from ramancarbon.dataio.jcamp import write_jcamp
from ramancarbon.echem.io import write_cv, write_eis, write_gcd
from ramancarbon.examples.demo_data import (
    make_cv_demo,
    make_demo,
    make_eis_demo,
    make_gcd_demo,
    make_xrd_demo,
)
from ramancarbon.xrd.io import write_pattern


def preparar(carpeta: Path) -> None:
    espectro = make_demo("MWCNT_FeSe")
    espectro.name = "M1"
    write_spectrum(espectro, carpeta / "m1.txt")
    write_jcamp(espectro.shift, espectro.intensity, carpeta / "m1.jdx",
                title="M1", laser_nm=532.0)
    write_pattern(make_xrd_demo("CNT_FeSe"), carpeta / "m1_drx.xy")
    write_cv(make_cv_demo("pseudocondensador"), carpeta / "m1_cv.txt")
    write_gcd(make_gcd_demo(), carpeta / "m1_gcd.txt")
    write_eis(make_eis_demo(), carpeta / "m1_eis.txt")
    (carpeta / "notas.md").write_text(
        "Muestra M1, sintetizada el 3 de septiembre.\n", encoding="utf-8")
    # El caso que de verdad pasa: la extensión miente.
    write_spectrum(make_demo("SWCNT"), carpeta / "m2.csv")


def main() -> None:
    carpeta = Path(tempfile.mkdtemp()) / "datos"
    carpeta.mkdir(parents=True)
    preparar(carpeta)

    print(f"Carpeta: {carpeta}\n")
    print("Qué es cada archivo, decidido por los números:\n")
    for archivo in sorted(carpeta.iterdir()):
        encontrado = detect(archivo)
        print(f"  {archivo.name:14s} {encontrado.kind:11s} "
              f"({encontrado.fmt}, {encontrado.confidence})")
        print(f"                 └ {encontrado.reasons[0]}")

    print("\nOrdenada por tipo:")
    for tipo, archivos in sorted(scan(carpeta).items()):
        print(f"  {tipo:12s} {', '.join(a.name for a in archivos)}")

    cargados, fallos = load_folder(carpeta)
    print(f"\nCargados {len(cargados)} de {len(cargados) + len(fallos)}:")
    for item in cargados:
        print(f"  {item.path.name:14s} → {type(item.data).__name__}")
    for ruta, error in fallos:
        print(f"  {ruta.name:14s} × {error.split(':', 1)[-1].strip()[:60]}…")
    print("  Un archivo que no es una medida no aborta la importación: la\n"
          "  carpeta con el LÉEME dentro es el caso normal, no el raro.\n")

    proyecto = Project(name="M1", notes="todo lo medido sobre esta muestra")
    for item in cargados:
        proyecto.add(item.data)
    proyecto.add_table(summary_table(
        [{"muestra": "M1", "técnica": item.kind,
          "puntos": len(proyecto.datasets[i].values)}
         for i, item in enumerate(cargados)],
        title="Qué hay en el proyecto",
    ))
    ruta = proyecto.save(carpeta.parent / "M1.rcproj")
    print(f"Proyecto: {ruta.name}, {ruta.stat().st_size / 1024:.0f} kB, "
          f"{len(proyecto.datasets)} medidas")

    reabierto = Project.load(ruta)
    print(f"Reabierto: «{reabierto.name}», {len(reabierto.datasets)} medidas, "
          f"{len(reabierto.results)} tabla(s)")
    for conjunto in reabierto.datasets:
        objeto = conjunto.to_object()
        print(f"  {conjunto.identifier:26s} {conjunto.kind:5s} "
              f"→ {type(objeto).__name__}  {list(conjunto.options)[:2]}")
    print("\n  Los ajustes viajan con los números: sin el láser, sin la\n"
          "  longitud de onda o sin la velocidad de barrido, los datos se\n"
          "  vuelven a abrir pero ya no se pueden analizar.")

    print(f"\nEs un ZIP corriente. Ábrelo con cualquier cosa:\n  {ruta}")


if __name__ == "__main__":
    main()
