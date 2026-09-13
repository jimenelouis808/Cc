"""Analizar un espectro y escribir el informe completo.

    python -m ramancarbon.examples.ex01_analizar
"""

from __future__ import annotations

from pathlib import Path

from ramancarbon import analyse
from ramancarbon.core.io import write_spectrum
from ramancarbon.examples.demo_data import make_demo


def main() -> None:
    out = Path("salida/ex01")

    # En tu caso esto sería:  spectrum = read_spectrum("muestra.txt", laser_nm=532)
    spectrum = make_demo("DWCNT", laser_nm=532.0, seed=0)
    write_spectrum(spectrum, out / "espectro_sintetico.txt")

    result = analyse(spectrum, basis="area")
    print(result.report())

    (out / "informe.txt").write_text(result.report(), encoding="utf-8")
    print(f"\nInforme guardado en {out / 'informe.txt'}")

    # Los números sueltos, para meterlos en una hoja de cálculo.
    print("\nResumen:")
    for key, value in result.to_dict().items():
        print(f"  {key:>18s}: {value}")


if __name__ == "__main__":
    main()
