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
    parser.add_argument(
        "--auto", action="store_true",
        help="elegir despiking, suavizado y rigidez de la línea base a partir "
             "del propio espectro, y explicar cada decisión. Lo que pongas a "
             "mano tiene prioridad sobre lo automático",
    )
    parser.add_argument(
        "--linea-base", dest="baseline", default="asls",
        choices=("asls", "arpls", "polynomial", "rubberband", "none"),
        help="método de línea base (por defecto asls; arpls no necesita "
             "parámetro de asimetría)",
    )
    parser.add_argument(
        "--lambda", dest="lam", type=float, default=None, metavar="VALOR",
        help="rigidez de la línea base. Sin este argumento se calcula a partir "
             "del espectro",
    )
    parser.add_argument(
        "--normalizar", default=None,
        choices=("0-100", "g", "max", "area", "minmax"),
        help="normalización opcional. Ojo: 0-100 y minmax restan un "
             "desplazamiento y eso SÍ cambia los cocientes; normaliza después "
             "de restar la línea base",
    )
    parser.add_argument(
        "--interferencias", action="store_true",
        help="buscar bandas que no son carbono (óxidos de catalizador, "
             "precursor de dopante sin reaccionar, sustrato) y excluir del "
             "cálculo de diámetros las que caigan en la ventana RBM. "
             "DESACTIVADO por defecto",
    )
    parser.add_argument(
        "--perfil", dest="profile", default=None,
        choices=("pseudo_voigt", "gaussian", "lorentzian", "bwf"),
        help="forzar un perfil en todas las componentes de la deconvolución. "
             "Sin este argumento se usan los perfiles por defecto de cada "
             "banda. pseudo_voigt deja que el ajuste decida la forma y "
             "devuelve η como resultado",
    )


def _preprocess_kwargs(args) -> dict:
    """Explicit preprocessing arguments, i.e. the ones the user actually set.

    Deliberately omits anything left at its default, so that ``--auto`` can
    fill those in and a flag given on the command line still wins.
    """
    kwargs: dict = {}
    if args.no_baseline:
        kwargs["baseline_method"] = None
    elif getattr(args, "baseline", "asls") != "asls" or not getattr(args, "auto", False):
        kwargs["baseline_method"] = (
            None if getattr(args, "baseline", "asls") == "none"
            else getattr(args, "baseline", "asls")
        )
    if getattr(args, "lam", None) is not None:
        kwargs["baseline_kwargs"] = {"lam": args.lam}
    if args.suavizado:
        kwargs["smooth_window"] = args.suavizado
    elif not getattr(args, "auto", False):
        kwargs["smooth_window"] = 0
    if getattr(args, "normalizar", None):
        kwargs["normalise_method"] = args.normalizar
    return kwargs


def _analyse_kwargs(args) -> dict:
    """Everything ``analyse`` needs from the shared options."""
    return {
        "basis": args.base,
        "rbm_parameterisation": args.rbm,
        "profile": getattr(args, "profile", None),
        "auto_preprocess": getattr(args, "auto", False),
        "check_interferences": getattr(args, "interferencias", False),
        "preprocess_kwargs": _preprocess_kwargs(args),
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
            control = analyse(_load(Path(args.control), args), **_analyse_kwargs(args))
        except (OSError, SpectrumReadError, ValueError) as exc:
            print(f"error leyendo el control: {exc}", file=sys.stderr)
            return 1

    result = analyse(
        spectrum, material_hint=args.material, control=control, **_analyse_kwargs(args)
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

    if args.exportar:
        from ..analysis.export import export_analysis

        files = export_analysis(result, Path(args.exportar), delimiter=args.separador)
        print("\nExportado:", file=sys.stderr)
        for path in files:
            print(f"  {path}", file=sys.stderr)
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

    from ..analysis.batch import analyse_many, summarise

    spectra = []
    failures: list[tuple[Path, str]] = []
    for path in files:
        try:
            spectra.append(read_spectrum(path, laser_nm=args.laser))
        except (OSError, ValueError) as exc:
            failures.append((path, str(exc)))
            print(f"  ✗ {path.name}: {exc}", file=sys.stderr)

    def tick(done: int, total: int) -> None:
        print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)

    results, analysis_failures = analyse_many(
        spectra,
        workers=args.procesos,
        progress=tick,
        material_hint=args.material,
        **_analyse_kwargs(args),
    )
    print(file=sys.stderr)
    failures.extend((Path(name), message) for name, message in analysis_failures)
    rows = [r.to_dict() for r in results]

    if not rows:
        print("error: no se ha podido analizar ningún espectro", file=sys.stderr)
        return 1

    if not args.sin_estadistica:
        print(summarise(results, failures=analysis_failures).summary(), file=sys.stderr)
        print(file=sys.stderr)

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

    from ..core.preprocess import auto_settings, preprocess

    settings = _preprocess_kwargs(args)
    if args.auto:
        chosen = auto_settings(spectrum)
        merged = chosen.to_kwargs()
        merged.update(settings)
        settings = merged
        print(chosen.summary(), file=sys.stderr)
        print(file=sys.stderr)
    processed, _ = preprocess(spectrum, **settings)

    if args.comparar:
        comparison = compare_models(processed, profile=args.profile)
        print(comparison.summary())
        print()
        best = comparison.results[comparison.best]
        print(best.summary())
        _maybe_export_fit(args, best)
        return 0

    try:
        model = build_model(
            processed, preset=args.modelo, metallic=args.metalico, profile=args.profile
        )
        result = fit_model(processed, model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Modelo: {PRESET_LABELS.get(args.modelo, args.modelo)}")
    print(f"Ventana: {model.window[0]:.0f}–{model.window[1]:.0f} cm⁻¹")
    print(f"Perfil: {args.profile or 'según la base de datos'}")
    print()
    print(result.summary())
    _maybe_export_fit(args, result)
    return 0


def _maybe_export_fit(args, fit) -> None:
    """Write the components and curves of a fit, if asked for."""
    if not args.exportar:
        return
    from ..analysis.export import export_components, export_curves

    out = Path(args.exportar)
    stem = Path(args.espectro).stem
    written = [
        export_components(fit, out / f"{stem}_componentes.csv", delimiter=args.separador),
        export_curves(fit, out / f"{stem}_curvas.csv", delimiter=args.separador),
    ]
    print("\nExportado:", file=sys.stderr)
    for path in written:
        print(f"  {path}", file=sys.stderr)


def cmd_laseres(args) -> int:
    """Combine measurements of the same sample at different excitations."""
    from ..analysis.multiwavelength import compare_excitations

    results = []
    for item in args.espectros:
        path = Path(item)
        try:
            spectrum = read_spectrum(path)
        except (OSError, SpectrumReadError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if spectrum.laser_nm is None:
            print(
                f"error: {path.name} no indica su longitud de onda y este "
                "comando la necesita para cada espectro. Ponla en la cabecera "
                "del archivo o renombra con el láser",
                file=sys.stderr,
            )
            return 1
        results.append(analyse(spectrum, **_analyse_kwargs(args)))

    try:
        combined = compare_excitations(results, tolerance=args.tolerancia)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(combined.summary())
    return 0


def cmd_calibrar(args) -> int:
    """Measure the axis offset from the silicon line, and optionally fix it."""
    from ..analysis.quality import SILICON_LINE, calibrate, check_quality, silicon_offset

    try:
        spectrum = read_spectrum(Path(args.espectro))
    except (OSError, SpectrumReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(check_quality(spectrum).summary())
    print()

    offset = args.offset if args.offset is not None else silicon_offset(spectrum)
    if offset is None:
        print(
            f"No se ha encontrado ninguna línea estrecha cerca de "
            f"{SILICON_LINE} cm⁻¹.\n"
            "Mide un patrón de silicio en la misma sesión, o pasa el "
            "desplazamiento con --offset si lo conoces por otra vía.",
            file=sys.stderr,
        )
        return 1

    print(f"Línea de referencia medida en {SILICON_LINE + offset:.2f} cm⁻¹")
    print(f"Desplazamiento del eje       : {offset:+.2f} cm⁻¹")
    if abs(offset) <= 1.0:
        print("El eje está bien calibrado.")
    else:
        print(
            f"\nRéstale {offset:+.2f} cm⁻¹ a todas las posiciones antes de "
            "interpretar desplazamientos: los efectos de dopado son de este "
            "mismo orden."
        )

    if args.corregir:
        written = write_spectrum(calibrate(spectrum, offset), Path(args.corregir))
        print(f"\nEspectro corregido guardado en {written}", file=sys.stderr)
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

    if args.dopantes:
        print("Firmas de dopado — desplazamientos esperados a 2.33 eV\n")
        for signature in db.dopants.values():
            print(f"{signature.key}  —  {signature.label}")
            print(f"  portadores : {signature.carrier}")
            print(f"  ΔG         : {signature.g_shift[0]:+g} a {signature.g_shift[1]:+g} cm⁻¹")
            print(f"  Δ2D        : {signature.d2_shift[0]:+g} a {signature.d2_shift[1]:+g} cm⁻¹")
            print(f"  I_D/I_G    : {signature.id_ig}")
            print(f"  hospedador : {', '.join(signature.host) or 'cualquiera'}")
            print(f"  confianza  : {signature.confidence}")
            print(f"  fuente     : {signature.source}")
            if signature.notes:
                print(f"  nota       : {signature.notes}")
            print()
        guidance = db.dopant_guidance
        if guidance:
            print("CÓMO LEER ESTO\n")
            for key in ("size_vs_charge", "practical_rule", "what_raman_cannot_do"):
                if key in guidance:
                    print(f"  {guidance[key]}\n")
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
            "  ramancarbon analizar m.txt --laser 532 --auto --perfil pseudo_voigt\n"
            "  ramancarbon analizar m.txt --laser 532 --exportar resultados/\n"
            "  ramancarbon laseres m_532.txt m_633.txt\n"
            "  ramancarbon lote datos/ --procesos 4 --csv r.csv\n"
            "  ramancarbon calibrar patron_si.txt\n"
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
    p.add_argument("--exportar", default=None, metavar="CARPETA",
                   help="exportar todo a una carpeta: informe, tabla de "
                        "componentes, curvas de la deconvolución, espectro "
                        "procesado y JSON completo")
    p.add_argument("--separador", default=",", metavar="CAR",
                   help="separador de columnas en los CSV (usa ';' si tu hoja "
                        "de cálculo está en configuración española)")
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
    p.add_argument("--separador", default=",", metavar="CAR",
                   help="separador de columnas del CSV")
    p.add_argument("--procesos", type=int, default=None, metavar="N",
                   help="procesos en paralelo. Por defecto, tantos como "
                        "núcleos. Usa 1 para depurar: un error dentro de un "
                        "proceso hijo es mucho más difícil de leer")
    p.add_argument("--sin-estadistica", action="store_true",
                   help="no imprimir el resumen estadístico del lote")
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
    p.add_argument("--exportar", default=None, metavar="CARPETA",
                   help="exportar la tabla de componentes y las curvas")
    p.add_argument("--separador", default=",", metavar="CAR",
                   help="separador de columnas de los CSV")
    _add_common(p)
    p.set_defaults(func=cmd_deconvolucionar)

    p = sub.add_parser(
        "laseres",
        help="combinar el mismo material medido a varias excitaciones",
        description=(
            "Mide la dispersión de cada banda a partir de dos o más láseres. "
            "Eso permite tres cosas imposibles con uno solo: distinguir una "
            "banda de doble resonancia de una impostora que no se desplaza, "
            "detectar carbono amorfo por la dispersión de la banda G (cero en "
            "grafito, 6–10 cm⁻¹/eV en amorfo), y comprobar que las dos "
            "excitaciones dan el mismo tamaño de cristalito."
        ),
    )
    p.add_argument("espectros", nargs="+",
                   help="dos o más archivos del MISMO material a láseres "
                        "distintos; cada uno debe indicar el suyo en la cabecera")
    p.add_argument("--tolerancia", type=float, default=15.0, metavar="CM1_EV",
                   help="margen en cm⁻¹/eV para dar por buena una dispersión "
                        "(por defecto 15, generoso a propósito: con dos láseres "
                        "la pendiente es una estimación de dos puntos)")
    _add_common(p)
    p.set_defaults(func=cmd_laseres)

    p = sub.add_parser(
        "calibrar",
        help="medir y corregir el desplazamiento del eje con la línea del silicio",
        description=(
            "El silicio tiene su fonón de primer orden en 520.7 cm⁻¹ exactos. "
            "Si tu muestra está sobre un sustrato de silicio y el láser llega "
            "a él, la desviación de esa línea es el error de calibración del "
            "equipo ese día — del mismo tamaño que los desplazamientos por "
            "dopado que luego quieres interpretar."
        ),
    )
    p.add_argument("espectro", help="archivo del espectro")
    p.add_argument("--corregir", default=None, metavar="ARCHIVO",
                   help="escribir el espectro con el eje corregido")
    p.add_argument("--offset", type=float, default=None, metavar="CM1",
                   help="desplazamiento a restar, si lo sabes por otra vía")
    p.set_defaults(func=cmd_calibrar)

    p = sub.add_parser("bd", help="consultar la base de datos de literatura")
    p.add_argument("--banda", default=None, metavar="CLAVE",
                   help="detalle de una banda (D, G, 2D, RBM…)")
    p.add_argument("--materiales", action="store_true",
                   help="listar los materiales de referencia")
    p.add_argument("--rbm", action="store_true",
                   help="listar las parametrizaciones RBM↔diámetro")
    p.add_argument("--dopantes", action="store_true",
                   help="listar las firmas de dopado (N, S, O, P, Se, B) y "
                        "cómo leerlas")
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
