"""Command line interface.

Five subcommands, matching the five things people actually do:

``analizar``
    One spectrum in, a full written report out.
``lote``
    A folder in, a CSV table out. The batch case.
``deconvolucionar``
    Just the D–G deconvolution, with model comparison.
``bd``
    Inspect the literature database — what the program believes and where
    each number came from.
``demo``
    Generate synthetic spectra to try things on.

Everything the GUI can do except the interactive editing is here, because a
tool that only works through a window cannot be scripted, and a batch of
three hundred map spectra is not a thing to click through.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..analysis.report import analyse
from ..core.io import TEXT_SUFFIXES, SpectrumReadError, read_spectrum, write_spectrum
from ..database import load_database
from ..models.deconvolution import PRESET_LABELS, PRESETS, build_model, compare_models
from ..models.fitting import fit_model


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--laser", type=float, default=None, metavar="NM",
        help="longitud de onda de excitación en nm (532, 633, 785…). "
             "Obligatoria si el archivo no la lleva en la cabecera: sin ella "
             "no se corrigen las posiciones por dispersión ni se puede "
             "calcular el tamaño de cristalito",
    )
    parser.add_argument(
        "--base", choices=("area", "height"), default="area",
        help="cocientes a partir de áreas integradas (por defecto) o de "
             "alturas de pico. Un I_D/I_G de áreas es 2–3 veces el de alturas "
             "para el mismo espectro: no los mezcles entre muestras",
    )
    parser.add_argument(
        "--rbm", default=None, metavar="CLAVE",
        help="parametrización RBM↔diámetro (ver «ramancarbon bd --rbm»). "
             "Por defecto la de haces/polvo",
    )
    parser.add_argument(
        "--sin-linea-base", action="store_true", dest="no_baseline",
        help="no restar línea base (los datos ya vienen corregidos)",
    )
    parser.add_argument(
        "--suavizado", type=int, default=0, metavar="PTS",
        help="ventana de Savitzky-Golay en puntos; 0 (por defecto) no suaviza",
    )


def _preprocess_kwargs(args) -> dict:
    return {
        "baseline_method": None if args.no_baseline else "asls",
        "smooth_window": args.suavizado,
    }


def _load(path: Path, args) -> "object":
    spectrum = read_spectrum(path, laser_nm=args.laser)
    if spectrum.laser_nm is None:
        print(
            f"aviso: {path.name} no indica la longitud de onda del láser y no "
            "se ha pasado --laser. El análisis continuará, pero las posiciones "
            "no se corrigen por dispersión y no habrá tamaño de cristalito.",
            file=sys.stderr,
        )
    return spectrum


# ----------------------------------------------------------------------
def cmd_analizar(args) -> int:
    """Analyse one spectrum and print the report."""
    path = Path(args.espectro)
    try:
        spectrum = _load(path, args)
    except (OSError, SpectrumReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    control = None
    if args.control:
        try:
            control = analyse(
                _load(Path(args.control), args),
                basis=args.base,
                preprocess_kwargs=_preprocess_kwargs(args),
            )
        except (OSError, SpectrumReadError, ValueError) as exc:
            print(f"error leyendo el control: {exc}", file=sys.stderr)
            return 1

    result = analyse(
        spectrum,
        basis=args.base,
        rbm_parameterisation=args.rbm,
        material_hint=args.material,
        control=control,
        preprocess_kwargs=_preprocess_kwargs(args),
    )
    report = result.report()
    print(report)

    if args.salida:
        out = Path(args.salida)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nInforme guardado en {out}", file=sys.stderr)

    if args.figura:
        _save_figure(result, Path(args.figura))
        print(f"Figura guardada en {args.figura}", file=sys.stderr)

    if args.procesado:
        written = write_spectrum(result.processed, Path(args.procesado))
        print(f"Espectro procesado guardado en {written}", file=sys.stderr)
    return 0


def _save_figure(result, path: Path) -> None:
    """Render the summary figure without needing a display."""
    import matplotlib

    matplotlib.use("Agg")
    from ..gui.plots import figure_for_report
    from ..gui.theme import LIGHT, matplotlib_style

    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(matplotlib_style(LIGHT)):
        figure = figure_for_report(result, LIGHT)
        figure.savefig(path)


def cmd_lote(args) -> int:
    """Analyse every spectrum in a folder and write a CSV table."""
    root = Path(args.carpeta)
    if not root.is_dir():
        print(f"error: {root} no es una carpeta", file=sys.stderr)
        return 1
    pattern = "**/*" if args.recursivo else "*"
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        print(
            f"error: no hay archivos {', '.join(TEXT_SUFFIXES)} en {root}",
            file=sys.stderr,
        )
        return 1

    rows: list[dict] = []
    failures: list[tuple[Path, str]] = []
    for path in files:
        try:
            spectrum = read_spectrum(path, laser_nm=args.laser)
            result = analyse(
                spectrum,
                basis=args.base,
                rbm_parameterisation=args.rbm,
                material_hint=args.material,
                preprocess_kwargs=_preprocess_kwargs(args),
            )
        except (OSError, ValueError) as exc:
            failures.append((path, str(exc)))
            print(f"  ✗ {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append(result.to_dict())
        print(
            f"  ✓ {path.name}: {result.classification.label} "
            f"(I_D/I_G = {result.id_ig:.3f})" if result.id_ig is not None
            else f"  ✓ {path.name}: {result.classification.label}",
            file=sys.stderr,
        )

    if not rows:
        print("error: no se ha podido analizar ningún espectro", file=sys.stderr)
        return 1

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_cell(row.get(column)) for column in columns))
    text = "\n".join(lines) + "\n"

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(
            f"\n{len(rows)} espectros analizados, {len(failures)} con error.\n"
            f"Tabla guardada en {out}",
            file=sys.stderr,
        )
    else:
        print(text)
    return 0 if not failures else 0


def _csv_cell(value) -> str:
    if value is None:
        return ""
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    if "," in text or '"' in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def cmd_deconvolucionar(args) -> int:
    """Fit the D–G region and print the components."""
    path = Path(args.espectro)
    try:
        spectrum = _load(path, args)
    except (OSError, SpectrumReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    from ..core.preprocess import preprocess

    processed, _ = preprocess(spectrum, **_preprocess_kwargs(args))

    if args.comparar:
        comparison = compare_models(processed)
        print(comparison.summary())
        print()
        print(comparison.results[comparison.best].summary())
        return 0

    try:
        model = build_model(processed, preset=args.modelo, metallic=args.metalico)
        result = fit_model(processed, model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Modelo: {PRESET_LABELS.get(args.modelo, args.modelo)}")
    print(f"Ventana: {model.window[0]:.0f}–{model.window[1]:.0f} cm⁻¹")
    print()
    print(result.summary())
    return 0


def cmd_bd(args) -> int:
    """Print the contents of the literature database."""
    db = load_database()
    if args.rbm:
        print("Parametrizaciones ω_RBM = A/d + B (d en nm, ω en cm⁻¹)\n")
        for key, param in db.rbm.items():
            marker = " (por defecto)" if key == db.rbm_default else ""
            print(f"{key}{marker}")
            print(f"  {param.label}")
            if param.is_multiplicative:
                print(f"  A = {param.A:g}, forma multiplicativa con "
                      f"C_e = {param.environment_correction:g} nm⁻²")
            else:
                print(f"  A = {param.A:g}, B = {param.B:g}")
            print(f"  entorno: {param.environment}")
            print(f"  válido para d = {param.diameter_range_nm[0]:g}–"
                  f"{param.diameter_range_nm[1]:g} nm")
            print(f"  confianza: {param.confidence}")
            print(f"  fuente: {param.source}")
            if param.notes:
                print(f"  nota: {param.notes}")
            print()
        return 0

    if args.materiales:
        print("Materiales de referencia\n")
        for material in db.materials.values():
            print(f"{material.key}  —  {material.label}")
            print(f"  familia: {material.family}, paredes: {material.walls}")
            print(f"  RBM: {material.rbm.get('expected', '—')}")
            for name, entry in material.bands.items():
                low, high = entry["position"]
                print(f"  {name:>7s}: {low:.0f}–{high:.0f} cm⁻¹")
            for name, span in material.ratios.items():
                print(f"  {name:>7s}: {span[0]:g}–{span[1]:g} ({material.intensity_basis})")
            print(f"  confianza: {material.confidence}")
            print(f"  fuente: {material.source}")
            if material.notes:
                print(f"  nota: {material.notes}")
            print()
        return 0

    if args.banda:
        try:
            band = db.band(args.banda)
        except Exception as exc:  # DatabaseError
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{band.key} — {band.name}\n")
        print(f"  posición (2.33 eV) : {band.position:.1f} cm⁻¹")
        print(f"  ventana            : {band.window[0]:.0f}–{band.window[1]:.0f} cm⁻¹")
        print(f"  dispersión         : {band.dispersion:+.1f} cm⁻¹/eV")
        if args.laser:
            from ..core.spectrum import laser_energy_ev

            ev = laser_energy_ev(args.laser)
            lo, hi = band.window_at(ev)
            print(f"  a {args.laser:g} nm         : {band.position_at(ev):.1f} cm⁻¹ "
                  f"(ventana {lo:.0f}–{hi:.0f})")
        print(f"  FWHM típica        : {band.typical_fwhm[0]:g}–{band.typical_fwhm[1]:g} cm⁻¹")
        print(f"  perfil por defecto : {band.default_profile}")
        print(f"  confianza          : {band.confidence}")
        print(f"\n  origen: {band.origin}")
        print(f"\n  notas: {band.notes}")
        print(f"\n  fuente: {band.source}")
        return 0

    print(db.summary())
    print()
    print("Bandas:")
    for band in sorted(db.bands.values(), key=lambda b: b.position):
        print(f"  {band.key:>8s}  {band.position:7.1f} cm⁻¹  "
              f"({band.window[0]:.0f}–{band.window[1]:.0f}), "
              f"dispersión {band.dispersion:+.0f} cm⁻¹/eV  [{band.confidence}]")
    print()
    print("Usa --banda CLAVE, --materiales o --rbm para más detalle.")
    return 0


def cmd_demo(args) -> int:
    """Write synthetic spectra to a folder."""
    from ..examples.demo_data import DEMO_KINDS, make_demo

    out = Path(args.carpeta)
    out.mkdir(parents=True, exist_ok=True)
    kinds = [args.material] if args.material else list(DEMO_KINDS)
    for index, kind in enumerate(kinds):
        try:
            spectrum = make_demo(kind, laser_nm=args.laser or 532.0, seed=index)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        path = write_spectrum(spectrum, out / f"{spectrum.name}.txt")
        print(f"  {path}")
    print(
        "\nSon espectros SINTÉTICOS, generados por el programa. Sirven para "
        "probarlo, no para validarlo contra la realidad.",
        file=sys.stderr,
    )
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ramancarbon",
        description=(
            "Análisis de espectros Raman de nanomateriales de carbono: "
            "identificación SWCNT/DWCNT/MWCNT, deconvolución de las bandas D y "
            "G, cocientes I_D/I_G, I_2D/I_G e I_D/I_D', diámetros por RBM y "
            "desplazamientos frente a la literatura."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  ramancarbon analizar muestra.txt --laser 532\n"
            "  ramancarbon analizar dopado.txt --laser 532 --control prístino.txt\n"
            "  ramancarbon lote datos/ --laser 633 --csv resultados.csv\n"
            "  ramancarbon deconvolucionar muestra.txt --comparar\n"
            "  ramancarbon bd --banda 2D --laser 785\n"
            "  ramancarbon demo salida/\n"
        ),
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("analizar", help="analizar un espectro y escribir el informe")
    p.add_argument("espectro", help="archivo del espectro (.txt, .csv, .dat…)")
    p.add_argument("--control", default=None, metavar="ARCHIVO",
                   help="espectro de referencia sin tratar, medido el mismo día. "
                        "Comparar contra él elimina la deriva del equipo y es "
                        "mucho más fiable que comparar contra la literatura")
    p.add_argument("--material", default=None, metavar="CLAVE",
                   help="forzar el material de referencia en vez de usar el "
                        "clasificador (ver «ramancarbon bd --materiales»)")
    p.add_argument("--salida", default=None, metavar="ARCHIVO",
                   help="guardar el informe en un archivo de texto")
    p.add_argument("--figura", default=None, metavar="ARCHIVO",
                   help="guardar la figura resumen (.png, .pdf, .svg)")
    p.add_argument("--procesado", default=None, metavar="ARCHIVO",
                   help="guardar el espectro ya preprocesado")
    _add_common(p)
    p.set_defaults(func=cmd_analizar)

    p = sub.add_parser("lote", help="analizar una carpeta entera y volcar una tabla")
    p.add_argument("carpeta", help="carpeta con los espectros")
    p.add_argument("--csv", default=None, metavar="ARCHIVO",
                   help="archivo CSV de salida; si se omite se imprime por pantalla")
    p.add_argument("--recursivo", action="store_true",
                   help="buscar también en subcarpetas")
    p.add_argument("--material", default=None, metavar="CLAVE",
                   help="forzar el material de referencia")
    _add_common(p)
    p.set_defaults(func=cmd_lote)

    p = sub.add_parser("deconvolucionar", help="ajustar la región D–G")
    p.add_argument("espectro", help="archivo del espectro")
    p.add_argument("--modelo", default="three_band", choices=list(PRESETS),
                   help="preajuste a usar (por defecto three_band: D + G + D')")
    p.add_argument("--comparar", action="store_true",
                   help="ajustar 2, 3, 4 y 5 bandas y elegir por criterio de "
                        "información en vez de por costumbre")
    p.add_argument("--metalico", action="store_true",
                   help="para el modelo swcnt_g: ajustar G⁻ con perfil "
                        "Breit-Wigner-Fano (tubos metálicos)")
    _add_common(p)
    p.set_defaults(func=cmd_deconvolucionar)

    p = sub.add_parser("bd", help="consultar la base de datos de literatura")
    p.add_argument("--banda", default=None, metavar="CLAVE",
                   help="detalle de una banda (D, G, 2D, RBM…)")
    p.add_argument("--materiales", action="store_true",
                   help="listar los materiales de referencia")
    p.add_argument("--rbm", action="store_true",
                   help="listar las parametrizaciones RBM↔diámetro")
    p.add_argument("--laser", type=float, default=None, metavar="NM",
                   help="mostrar las posiciones corregidas a este láser")
    p.set_defaults(func=cmd_bd)

    p = sub.add_parser("demo", help="generar espectros sintéticos de prueba")
    p.add_argument("carpeta", help="carpeta donde escribirlos")
    p.add_argument("--material", default=None, metavar="CLAVE",
                   help="generar solo este material")
    p.add_argument("--laser", type=float, default=None, metavar="NM",
                   help="longitud de onda de excitación (por defecto 532)")
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``ramancarbon`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrumpido.", file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
