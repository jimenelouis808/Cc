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
        "n_d": getattr(args, "n_d", None),
        "n_g": getattr(args, "n_g", None),
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
        if args.n_d or args.n_g:
            from ..models.deconvolution import build_region_model

            model = build_region_model(
                processed,
                n_d=args.n_d or 1,
                n_g=args.n_g or 1,
                profile=args.profile or "pseudo_voigt",
                metallic=args.metalico,
            )
        else:
            model = build_model(
                processed, preset=args.modelo, metallic=args.metalico,
                profile=args.profile,
            )
        result = fit_model(processed, model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Modelo: {model.name if (args.n_d or args.n_g) else PRESET_LABELS.get(args.modelo, args.modelo)}")
    print(f"Ventana: {model.window[0]:.0f}–{model.window[1]:.0f} cm⁻¹")
    effective = args.profile or ("pseudo_voigt" if (args.n_d or args.n_g) else None)
    print(f"Perfil: {effective or 'según la base de datos'}")
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


def cmd_tmd(args) -> int:
    """Analyse a dichalcogenide spectrum."""
    from ..analysis.tmd import analyse_tmd, tmd_materials

    if args.listar:
        for material in tmd_materials():
            print(f"{material.key}  —  {material.label}")
            for mode in material.modes.values():
                print(f"    {mode.key:<9s} {mode.position:7.1f} cm⁻¹   {mode.label}")
            if material.separation_by_layers:
                spans = ", ".join(
                    f"{k}: {v[0]:g}–{v[1]:g}"
                    for k, v in material.separation_by_layers.items()
                )
                print(f"    separación E₂g–A₁g por capas → {spans}")
            else:
                print("    los dos modos son casi degenerados: no cuenta capas")
            print(f"    confianza: {material.confidence}")
            print(f"    nota: {material.notes}")
            print()
        return 0

    if not args.espectro:
        print("error: indica un archivo de espectro, o usa --listar",
              file=sys.stderr)
        return 1
    try:
        spectrum = read_spectrum(Path(args.espectro), laser_nm=args.laser)
    except (OSError, SpectrumReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = analyse_tmd(spectrum, material=args.material)
    print(result.summary())
    if result.fit is not None and args.exportar:
        from ..analysis.export import export_components, export_curves

        out = Path(args.exportar)
        stem = Path(args.espectro).stem
        for path in (
            export_components(result.fit, out / f"{stem}_componentes.csv"),
            export_curves(result.fit, out / f"{stem}_curvas.csv"),
        ):
            print(f"  {path}", file=sys.stderr)
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
    from ..examples.demo_data import DEMO_KINDS, make_demo, make_tmd_demo, TMD_DEMOS

    out = Path(args.carpeta)
    out.mkdir(parents=True, exist_ok=True)

    if args.tmd:
        for index, (material, layers) in enumerate(TMD_DEMOS):
            spectrum = make_tmd_demo(
                material, layers, laser_nm=args.laser or 532.0, seed=index
            )
            print(f"  {write_spectrum(spectrum, out / f'{spectrum.name}.txt')}")
        print(
            "\nSon espectros SINTÉTICOS de dicalcogenuros. Pruébalos con "
            "«ramancarbon tmd».",
            file=sys.stderr,
        )
        return 0

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
# ----------------------------------------------------------------------
# diffraction
# ----------------------------------------------------------------------
def cmd_drx(args) -> int:
    """Identify phases in a diffractogram and, optionally, refine them."""
    from ..xrd.io import read_pattern
    from ..xrd.reference import describe_library, library_crystals
    from ..xrd.report import analyse_pattern

    if args.biblioteca:
        print(describe_library(args.cif or None))
        return 0

    pattern = read_pattern(
        args.patron,
        wavelength=args.longitud,
        anode=args.anodo,
        counts=not args.sin_cuentas,
        kalpha2_ratio=args.kalfa2,
    )
    candidates = None
    if args.fases:
        candidates = library_crystals(args.cif or None, only=args.fases)
        if not candidates:
            print(f"error: ninguna de las fases {args.fases} está en la biblioteca",
                  file=sys.stderr)
            return 1

    axis = None
    if args.textura:
        text = args.textura.replace(",", " ").split()
        if len(text) == 1 and len(text[0]) == 3:
            text = list(text[0])
        try:
            axis = tuple(int(v) for v in text)
        except ValueError:
            print(f"error: eje de textura ilegible: {args.textura!r}", file=sys.stderr)
            return 1
        if len(axis) != 3:
            print("error: el eje de textura necesita tres índices", file=sys.stderr)
            return 1

    result = analyse_pattern(
        pattern,
        candidates=candidates,
        extra_directories=args.cif or None,
        refine=not args.sin_refinar,
        max_phases=args.max_fases,
        preferred_axis=axis,
        instrument_fwhm=args.resolucion,
    )
    text = result.report(verbose=not args.breve)
    print(text)

    if args.salida:
        destination = Path(args.salida)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"\nInforme escrito en {destination}")
    if args.figura and result.refinement is not None:
        from ..gui.plots_xrd import figure_for_report
        from ..gui.theme import LIGHT

        path = Path(args.figura)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure_for_report(result.refinement, LIGHT).savefig(path, dpi=200)
        print(f"Figura escrita en {path}")
    if args.calculado and result.refinement is not None:
        path = Path(args.calculado)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# 2theta  observado  calculado  diferencia  fondo"]
        refinement = result.refinement
        for angle, observed, calculated, difference, background in zip(
            pattern.two_theta, pattern.intensity, refinement.calculated,
            refinement.difference, refinement.background,
        ):
            lines.append(
                f"{angle:10.5f} {observed:14.4f} {calculated:14.4f} "
                f"{difference:14.4f} {background:14.4f}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Patrón calculado escrito en {path}")
    return 0


def cmd_drx_lote(args) -> int:
    """Analyse a folder of diffractograms and dump one CSV row each."""
    from ..xrd.io import read_pattern
    from ..xrd.report import analyse_pattern

    folder = Path(args.carpeta)
    if not folder.is_dir():
        print(f"error: {folder} no es una carpeta", file=sys.stderr)
        return 1
    suffixes = {".xy", ".xye", ".dat", ".txt", ".asc", ".csv", ".xrdml", ".uxd"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        print(f"error: no hay difractogramas en {folder}", file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        try:
            pattern = read_pattern(path, wavelength=args.longitud, anode=args.anodo)
            result = analyse_pattern(
                pattern, extra_directories=args.cif or None,
                refine=not args.sin_refinar,
            )
        except (OSError, ValueError) as exc:
            print(f"  ✗ {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append(result.to_dict())
        print(f"  ✓ {path.name}: {rows[-1].get('fases') or 'sin fases'}")

    if not rows:
        print("error: no se ha podido analizar ningún difractograma", file=sys.stderr)
        return 1
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    destination = Path(args.csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(_csv_cell(row.get(c)) for c in columns) + "\n")
    print(f"\n{len(rows)} difractograma(s) → {destination}")
    return 0


# ----------------------------------------------------------------------
# electrochemistry
# ----------------------------------------------------------------------
def cmd_echem(args) -> int:
    """Analyse the electrochemistry of one electrode."""
    from ..echem.curve import Electrode
    from ..echem.io import read_cv, read_eis, read_gcd
    from ..echem.report import analyse_sample

    electrode = Electrode(
        mass_mg=args.masa,
        area_cm2=args.area,
        reference=args.referencia,
        ph=args.ph,
        resistance_ohm=args.resistencia,
        label=args.nombre,
    )
    rate = args.velocidad / 1000.0 if args.velocidad else None

    cv = read_cv(args.cv, scan_rate=rate, electrode=electrode) if args.cv else None
    series = [
        read_cv(path, electrode=electrode)
        for path in (args.velocidades or [])
    ]
    gcd = (
        read_gcd(args.gcd, electrode=electrode,
                 current=args.corriente / 1000.0 if args.corriente else None)
        if args.gcd else None
    )
    eis = read_eis(args.eis, electrode=electrode) if args.eis else None
    lsv = read_cv(args.polarizacion, scan_rate=0.005, electrode=electrode) \
        if args.polarizacion else None

    if not any((cv, series, gcd, eis, lsv)):
        print(
            "error: no se ha dado ninguna medida. Usa --cv, --gcd, --eis, "
            "--velocidades o --polarizacion",
            file=sys.stderr,
        )
        return 1

    result = analyse_sample(
        name=args.nombre,
        cv=cv,
        rate_series=series or None,
        gcd=gcd,
        eis=eis,
        catalysis_curve=lsv,
        reaction=args.reaccion,
        circuit=args.circuito,
        non_faradaic=args.sin_faradaica,
    )
    text = result.report(verbose=not args.breve)
    print(text)

    if args.salida:
        destination = Path(args.salida)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"\nInforme escrito en {destination}")
    if args.csv:
        row = result.to_dict()
        destination = Path(args.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            ",".join(row) + "\n" + ",".join(_csv_cell(v) for v in row.values()) + "\n",
            encoding="utf-8",
        )
        print(f"Tabla escrita en {destination}")
    return 0


def cmd_demo_datos(args) -> int:
    """Write synthetic diffractograms or electrochemistry files."""
    folder = Path(args.carpeta)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.tipo == "drx":
        from ..examples.demo_data import xrd_demo_spectra
        from ..xrd.io import write_pattern

        for pattern in xrd_demo_spectra(seed=3):
            written.append(write_pattern(pattern, folder / f"{pattern.name}.xye"))
    else:
        from ..echem.io import write_cv, write_eis, write_gcd
        from ..examples.demo_data import (
            ECHEM_DEMOS,
            cv_rate_series,
            make_eis_demo,
            make_gcd_demo,
            make_lsv_demo,
        )

        for kind in ECHEM_DEMOS:
            for curve in cv_rate_series(kind, seed=2):
                written.append(write_cv(curve, folder / f"{curve.name}.txt"))
            gcd = make_gcd_demo(kind, seed=1)
            written.append(write_gcd(gcd, folder / f"{gcd.name}.txt"))
        spectrum = make_eis_demo("R0-(R1|Q1)-Q2", seed=3)
        written.append(write_eis(spectrum, folder / "demo_eis.txt"))
        lsv = make_lsv_demo("OER", seed=1)
        written.append(write_cv(lsv, folder / f"{lsv.name}.txt"))

    for path in written:
        print(f"  {path}")
    print(f"\n{len(written)} archivo(s) escritos en {folder}")
    print(
        "\nSon datos SINTÉTICOS, calculados de la física que se quiere probar. "
        "No son medidas."
    )
    return 0


def cmd_mapa(args) -> int:
    """Read a Raman map and report what it contains."""
    import numpy as np

    from ..mapping import (
        band_position,
        band_ratio,
        coverage,
        despike_map,
        kmeans,
        mcr_als,
        pca,
        read_map,
        suggested_components,
    )

    cube = read_map(args.archivo, laser_nm=args.laser, spot_um=args.punto)
    print(cube.describe())
    warning = cube.sampling_warning()
    if warning:
        print(f"  aviso: {warning}")

    clean, replaced = despike_map(cube)
    print(f"\nRayos cósmicos: {replaced} canales sustituidos de "
          f"{cube.intensity.size}")

    low, high = args.banda
    mask, fraction = coverage(clean, low, high)
    print(f"Cobertura (S/R ≥ 5) en {low:g}–{high:g} cm⁻¹: "
          f"{100 * fraction:.0f} % del mapa")
    for note in mask.warnings:
        print(f"  aviso: {note}")

    position = band_position(clean, low, high)
    print(f"\n{position.describe()}")

    if args.cociente:
        (a, b), (c, d) = args.cociente[:2], args.cociente[2:]
        ratio = band_ratio(clean, (a, b), (c, d))
        print(f"{ratio.describe()}")
        for note in ratio.warnings:
            print(f"  aviso: {note}")

    components = pca(clean, args.componentes)
    print(f"\n{components.describe()}")
    print("  varianza: " + ", ".join(
        f"{100 * v:.1f} %" for v in components.explained[:args.componentes]))
    suggested = suggested_components(components)
    print(f"  componentes con más que ruido: {suggested}")

    groups = kmeans(clean, max(2, min(suggested, args.grupos)))
    print(f"\n{groups.describe()}")
    for note in groups.warnings:
        print(f"  aviso: {note}")

    if args.mcr:
        resolved = mcr_als(clean, max(2, suggested))
        print(f"\n{resolved.describe()}")
        for index in range(resolved.k):
            spectrum = resolved.component_spectrum(index)
            peak = spectrum.shift[int(np.argmax(spectrum.intensity))]
            print(f"  componente {index + 1}: máximo en {peak:.0f} cm⁻¹")
        for note in resolved.warnings[:2]:
            print(f"  aviso: {note}")

    if args.figura:
        from ..plotting import AxisStyle, Plot, Series, preset

        plot = Plot(name="mapa")
        for index in range(groups.k):
            centre = groups.centre_spectrum(index)
            plot.add(Series(x=centre.shift, y=centre.intensity,
                            label=f"grupo {index + 1}"))
        plot.style = preset(args.preajuste).replace(
            x=AxisStyle(label="Desplazamiento Raman (cm⁻¹)"),
            y=AxisStyle(label="Intensidad (u.a.)"),
            normalise="max", offset=0.15,
        )
        print(f"\nFigura: {plot.save(args.figura)}")
    return 0


def cmd_figura(args) -> int:
    """Draw one figure from several measurements, with the plot engine."""
    from ..dataio import load
    from ..plotting import AxisStyle, Plot, Series, preset

    plot = Plot(name=Path(args.salida).stem)
    labels: list[str] = []
    kinds: set[str] = set()
    for path in args.archivos:
        loaded = load(path, laser_nm=args.laser)
        table_x, table_y, label = _series_of(loaded)
        plot.add(Series(x=table_x, y=table_y, label=label))
        labels.append(label)
        kinds.add(loaded.kind)

    if len(kinds) > 1:
        print("error: no se puede dibujar en la misma figura "
              + ", ".join(sorted(kinds)) + ": los ejes no son los mismos",
              file=sys.stderr)
        return 1

    kind = kinds.pop()
    axes = {
        "raman": ("Desplazamiento Raman (cm⁻¹)", "Intensidad (u.a.)"),
        "xrd": ("2θ (°)", "Intensidad (cuentas)"),
        "cv": ("Potencial (V)", "Corriente (A)"),
        "gcd": ("Tiempo (s)", "Potencial (V)"),
        "eis": ("Z′ (Ω)", "−Z″ (Ω)"),
    }[kind]
    plot.style = preset(args.preajuste).replace(
        x=AxisStyle(label=axes[0], limits=tuple(args.limites) if args.limites else None),
        y=AxisStyle(label=axes[1], scale=args.escala),
        normalise=args.normalizar, offset=args.desplazar,
    )
    for position in args.marcar or ():
        plot.mark(position)

    written = plot.save(args.salida)
    print(f"{written}  ({len(labels)} series: {', '.join(labels)})")
    if args.datos:
        print(f"{plot.save_data(args.datos)}  (los números dibujados)")
    return 0


def _series_of(loaded):
    """The two columns to draw for any measurement, and a label."""
    import numpy as np

    data = loaded.data
    name = getattr(data, "name", loaded.path.stem)
    if hasattr(data, "shift"):
        return data.shift, data.intensity, name
    if hasattr(data, "two_theta"):
        return data.two_theta, data.intensity, name
    if hasattr(data, "scan_rate"):
        return data.potential, data.current, name
    if hasattr(data, "frequency"):
        return np.asarray(data.z.real), -np.asarray(data.z.imag), name
    return data.time, data.potential, name


def cmd_exportar(args) -> int:
    """Convert a measurement into any of the offered formats."""
    from ..dataio import load
    from ..dataio.export import export

    loaded = load(args.archivo, laser_nm=args.laser)
    for note in loaded.warnings:
        print(f"aviso: {note}")
    written = export(loaded.data, args.salida)
    print(f"{loaded.path.name} ({loaded.kind}) → {written}")
    return 0


def cmd_proyecto(args) -> int:
    """Build a project file from a folder, or list what one contains."""
    from ..dataio import Project, load_folder

    if args.accion == "crear":
        loaded, failures = load_folder(args.origen, laser_nm=args.laser)
        project = Project(name=Path(args.origen).resolve().name,
                          notes=args.notas or "")
        for item in loaded:
            project.add(item.data)
        written = project.save(args.destino)
        print(f"{written}  ({len(project.datasets)} medidas, "
              f"{written.stat().st_size / 1024:.0f} kB)")
        for path, message in failures:
            print(f"  omitido {path.name}: {message.split(':', 1)[-1].strip()[:70]}")
        return 0

    project = Project.load(args.origen)
    print(f"{project.name}  ({project.application_version or 'versión desconocida'}, "
          f"{project.modified or 'sin fecha'})")
    if project.notes:
        print(project.notes)
    for dataset in project.datasets:
        points = 0 if dataset.values is None else len(dataset.values)
        print(f"  {dataset.identifier:<28s} {dataset.kind:<6s} {points:6d} puntos"
              f"   {', '.join(sorted(dataset.options))}")
    if project.figures:
        print(f"  {len(project.figures)} figura(s)")
    for table in project.results:
        print(f"  tabla «{table.get('titulo', '')}»: "
              f"{len(table.get('filas', []))} filas")
    return 0


def cmd_micro(args) -> int:
    """Size, strain and carbon microstructure from a diffractogram."""
    from ..xrd.io import read_pattern
    from ..xrd.microstructure import (
        agreement,
        carbon_microstructure,
        compare_methods,
    )
    from ..xrd.search import find_peaks

    pattern = read_pattern(args.patron, anode=args.anodo)
    peaks = [p for p in find_peaks(pattern) if p.fwhm]
    if len(peaks) < 3:
        print(f"error: solo {len(peaks)} reflexiones con anchura medible; "
              "hacen falta tres para separar tamaño de deformación",
              file=sys.stderr)
        return 1

    print(f"{pattern.name}: {len(peaks)} reflexiones\n")
    angles = [p.two_theta for p in peaks]
    widths = [p.fwhm for p in peaks]
    results = compare_methods(angles, widths, pattern.wavelength,
                              instrument_fwhm=args.instrumento)
    for result in results:
        print(f"  {result.describe()}")
        for note in result.warnings[:2]:
            print(f"      · {note}")
    print(f"\n  → {agreement(results)}")

    if args.carbono:
        d002 = min(peaks, key=lambda p: abs(p.two_theta - 26.5))
        hundred = min(peaks, key=lambda p: abs(p.two_theta - 43.0))
        carbon = carbon_microstructure(
            d002.two_theta, d002.fwhm, hundred.two_theta, hundred.fwhm,
            wavelength=pattern.wavelength, instrument_fwhm=args.instrumento)
        print(f"\nCarbono: {carbon.describe()}")
        for note in carbon.warnings:
            print(f"  aviso: {note}")
    return 0


def cmd_tiempos(args) -> int:
    """Time every operation of the suite."""
    from ..benchmarks import main as benchmark_main

    arguments = []
    if args.rapido:
        arguments.append("--rapido")
    if args.json:
        arguments += ["--json", args.json]
    arguments += ["--repeticiones", str(args.repeticiones)]
    return benchmark_main(arguments)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ramancarbon",
        description=(
            "Caracterización de nanomateriales: Raman de carbono y de "
            "dicalcogenuros, mapas Raman con quimiometría, difracción de rayos "
            "X con identificación de fases, Rietveld, Le Bail y "
            "microestructura, y electroquímica (CV, carga-descarga, "
            "impedancia con circuitos y DRT, dQ/dV, HER/OER). Con motor de "
            "figuras configurable, importación y exportación universales y "
            "archivos de proyecto."
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
            "  ramancarbon deconvolucionar m.txt --picos-d 3 --picos-g 2\n"
            "  ramancarbon tmd mos2.txt\n"
            "  ramancarbon bd --banda 2D --laser 785\n"
            "  ramancarbon demo salida/\n"
            "\n"
            "  ramancarbon drx patron.xy --cif mis_cifs/ --textura 001\n"
            "  ramancarbon drx patron.xy --sin-refinar --breve\n"
            "  ramancarbon drx --biblioteca\n"
            "  ramancarbon drx-lote datos/ --csv fases.csv\n"
            "\n"
            "  ramancarbon echem --cv cv.txt --velocidad 20 --masa 2 --area 1\n"
            "  ramancarbon echem --gcd gcd.txt --masa 2 --eis eis.txt\n"
            "  ramancarbon echem --polarizacion lsv.txt --reaccion OER \\\n"
            "                    --ph 14 --area 1 --resistencia 3\n"
            "  ramancarbon demo-datos drx prueba_drx/\n"
            "\n"
            "  ramancarbon mapa mapa.txt --punto 1 --cociente 1280 1420 1500 1660\n"
            "  ramancarbon figura a.txt b.txt --salida fig.png --preajuste acs \\\n"
            "                     --normalizar max --desplazar 0.2\n"
            "  ramancarbon exportar espectro.txt espectro.jdx\n"
            "  ramancarbon proyecto crear datos/ sesion.rcproj\n"
            "  ramancarbon micro patron.xy --instrumento 0.08 --carbono\n"
            "  ramancarbon tiempos --rapido\n"
        ),
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("analizar", help="analizar un espectro y escribir el informe")
    p.add_argument("--picos-d", dest="n_d", type=int, default=None, metavar="N",
                   help="número de componentes en la región D, imponiendo el "
                        "modelo en vez de dejar que se elija entre preajustes")
    p.add_argument("--picos-g", dest="n_g", type=int, default=None, metavar="N",
                   help="número de componentes en la región G. Con cualquiera "
                        "de los dos, no se comparan modelos: pediste uno "
                        "concreto, así que ese es el que se usa")
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
    p.add_argument("--picos-d", dest="n_d", type=int, default=None, metavar="N",
                   help="número de componentes en la región D. Con --picos-g "
                        "ignora --modelo y construye el modelo a medida. Las "
                        "tres primeras son D, D3 y D4 (bandas con nombre); a "
                        "partir de ahí salen sin nombre y sin interpretación")
    p.add_argument("--picos-g", dest="n_g", type=int, default=None, metavar="N",
                   help="número de componentes en la región G. Las tres "
                        "primeras son G, D' y G⁻")
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

    p = sub.add_parser(
        "tmd",
        help="analizar dicalcogenuros (MoS₂, WS₂, MoSe₂, WSe₂, MoTe₂)",
        description=(
            "Cuenta capas por la SEPARACIÓN entre el modo E₂g (en el plano) y "
            "el A₁g (fuera del plano), que crece de forma monótona al apilar. "
            "Al ser una diferencia, cualquier error común de calibración se "
            "cancela. Detecta también la fase 1T′ metálica por sus modos J."
        ),
    )
    p.add_argument("espectro", nargs="?", help="archivo del espectro")
    p.add_argument("--material", default=None, metavar="CLAVE",
                   help="forzar el material en vez de identificarlo")
    p.add_argument("--laser", type=float, default=None, metavar="NM",
                   help="longitud de onda de excitación")
    p.add_argument("--listar", action="store_true",
                   help="listar los TMD de la base de datos y sus modos")
    p.add_argument("--exportar", default=None, metavar="CARPETA",
                   help="exportar componentes y curvas del ajuste")
    p.set_defaults(func=cmd_tmd)

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
    p.add_argument("--tmd", action="store_true",
                   help="generar espectros de dicalcogenuros en vez de carbono")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser(
        "drx", help="identificar fases en un difractograma y refinar (Rietveld)"
    )
    p.add_argument("patron", nargs="?", default=None,
                   help="difractograma (.xy, .xye, .dat, .txt, .xrdml, .uxd)")
    p.add_argument("--anodo", default="Cu", metavar="ELEMENTO",
                   help="ánodo del tubo: Cu, Co, Fe, Mo, Cr, Ag (por defecto Cu)")
    p.add_argument("--longitud", type=float, default=None, metavar="A",
                   help="longitud de onda Kα₁ en Å; anula la del archivo. Una λ "
                        "equivocada escala TODA la celda por el mismo factor y "
                        "nada en el ajuste protesta")
    p.add_argument("--kalfa2", type=float, default=0.5, metavar="R",
                   help="intensidad de Kα₂ respecto a Kα₁ (0.5 por defecto; "
                        "pon 0 con monocromador o en sincrotrón)")
    p.add_argument("--sin-cuentas", action="store_true",
                   help="los datos NO son cuentas crudas (ya restados o "
                        "escalados), así que √N no es su incertidumbre")
    p.add_argument("--cif", action="append", default=None, metavar="CARPETA",
                   help="carpeta con CIF de referencia; repetible. Descárgalos "
                        "de la Crystallography Open Database")
    p.add_argument("--fases", nargs="+", default=None, metavar="NOMBRE",
                   help="probar solo estas fases de la biblioteca")
    p.add_argument("--max-fases", type=int, default=4, metavar="N",
                   help="número máximo de fases a aceptar (4 por defecto)")
    p.add_argument("--textura", default=None, metavar="HKL",
                   help="eje de orientación preferente, p. ej. 001. Sin él no "
                        "se refina textura, y en un material laminar acaba "
                        "absorbida por el U_iso")
    p.add_argument("--resolucion", type=float, default=0.06, metavar="GRADOS",
                   help="FWHM instrumental para Scherrer (0.06 por defecto). "
                        "Mídela con un patrón; un tamaño citado contra una "
                        "resolución supuesta es la suposición")
    p.add_argument("--sin-refinar", action="store_true",
                   help="solo identificar fases, sin Rietveld")
    p.add_argument("--biblioteca", action="store_true",
                   help="listar las fases de referencia disponibles y salir")
    p.add_argument("--salida", default=None, metavar="ARCHIVO",
                   help="escribir el informe en un archivo")
    p.add_argument("--figura", default=None, metavar="PNG",
                   help="guardar el gráfico de Rietveld con su diferencia")
    p.add_argument("--calculado", default=None, metavar="ARCHIVO",
                   help="volcar observado, calculado, diferencia y fondo")
    p.add_argument("--breve", action="store_true", help="omitir los avisos")
    p.set_defaults(func=cmd_drx)

    p = sub.add_parser("drx-lote", help="analizar una carpeta de difractogramas")
    p.add_argument("carpeta", help="carpeta con los difractogramas")
    p.add_argument("--csv", required=True, metavar="ARCHIVO",
                   help="archivo CSV de salida")
    p.add_argument("--anodo", default="Cu", metavar="ELEMENTO")
    p.add_argument("--longitud", type=float, default=None, metavar="A")
    p.add_argument("--cif", action="append", default=None, metavar="CARPETA")
    p.add_argument("--sin-refinar", action="store_true")
    p.set_defaults(func=cmd_drx_lote)

    p = sub.add_parser(
        "echem", help="analizar electroquímica: CV, carga-descarga, impedancia"
    )
    p.add_argument("--nombre", default="muestra", metavar="TEXTO")
    p.add_argument("--cv", default=None, metavar="ARCHIVO",
                   help="voltamperograma cíclico")
    p.add_argument("--velocidades", nargs="+", default=None, metavar="ARCHIVO",
                   help="serie de voltamperogramas a distintas velocidades, "
                        "para b, Trasatti y ECSA")
    p.add_argument("--gcd", default=None, metavar="ARCHIVO",
                   help="curva de carga-descarga galvanostática")
    p.add_argument("--eis", default=None, metavar="ARCHIVO",
                   help="espectro de impedancia")
    p.add_argument("--polarizacion", default=None, metavar="ARCHIVO",
                   help="curva de polarización para HER u OER")
    p.add_argument("--masa", type=float, default=None, metavar="MG",
                   help="masa de material ACTIVO en mg, no la del electrodo")
    p.add_argument("--area", type=float, default=None, metavar="CM2",
                   help="área geométrica en cm²")
    p.add_argument("--referencia", default="Ag/AgCl_3M", metavar="CLAVE",
                   help="electrodo de referencia (Ag/AgCl_3M, SCE, Hg/HgO_1M, "
                        "RHE…). Di siempre el relleno: Ag/AgCl 3 M y saturado "
                        "están a 13 mV")
    p.add_argument("--ph", type=float, default=None, metavar="PH",
                   help="pH del electrolito; hace falta para pasar a RHE")
    p.add_argument("--resistencia", type=float, default=None, metavar="OHM",
                   help="resistencia no compensada, para la corrección óhmica")
    p.add_argument("--velocidad", type=float, default=None, metavar="MV_S",
                   help="velocidad de barrido del CV en mV/s, si el archivo no "
                        "la declara")
    p.add_argument("--corriente", type=float, default=None, metavar="MA",
                   help="corriente del GCD en mA, si el archivo no la trae")
    p.add_argument("--reaccion", default="OER", choices=["OER", "HER"],
                   help="reacción de la curva de polarización")
    p.add_argument("--circuito", default="randles_cpe", metavar="CIRCUITO",
                   help="circuito equivalente, por nombre (randles, randles_cpe, "
                        "supercondensador, bateria…) o en notación R0-(R1|Q1)")
    p.add_argument("--sin-faradaica", action="store_true",
                   help="declarar que la ventana del CV está libre de corriente "
                        "faradaica, que es lo que hace válidos el C_dl y el ECSA")
    p.add_argument("--salida", default=None, metavar="ARCHIVO",
                   help="escribir el informe en un archivo")
    p.add_argument("--csv", default=None, metavar="ARCHIVO",
                   help="volcar las cifras principales como una fila CSV")
    p.add_argument("--breve", action="store_true", help="omitir los avisos")
    p.set_defaults(func=cmd_echem)

    p = sub.add_parser(
        "mapa", help="leer un mapa Raman y decir qué hay en él"
    )
    p.add_argument("archivo", help="archivo del mapa (texto, disposición larga o ancha)")
    p.add_argument("--laser", type=float, default=None, metavar="NM")
    p.add_argument("--punto", type=float, default=None, metavar="UM",
                   help="diámetro del punto láser. Sin él no se puede decir si "
                        "los píxeles vecinos son medidas independientes")
    p.add_argument("--banda", nargs=2, type=float, default=[1500.0, 1660.0],
                   metavar=("BAJO", "ALTO"),
                   help="ventana de la banda de referencia (por defecto la G)")
    p.add_argument("--cociente", nargs=4, type=float, default=None,
                   metavar=("A1", "A2", "B1", "B2"),
                   help="dos ventanas para un mapa de cocientes, p. ej. "
                        "1280 1420 1500 1660 para I_D/I_G")
    p.add_argument("--componentes", type=int, default=5,
                   help="componentes principales a calcular")
    p.add_argument("--grupos", type=int, default=3, help="grupos de k-medias")
    p.add_argument("--mcr", action="store_true",
                   help="resolver también las componentes con MCR-ALS")
    p.add_argument("--figura", default=None, metavar="ARCHIVO",
                   help="dibujar los espectros medios de cada grupo")
    p.add_argument("--preajuste", default="predeterminado",
                   help="preajuste de figura (ver «ramancarbon figura --ayuda-preajustes»)")
    p.set_defaults(func=cmd_mapa)

    p = sub.add_parser(
        "figura", help="dibujar una o varias medidas con el motor de figuras"
    )
    p.add_argument("archivos", nargs="+", help="medidas del mismo tipo")
    p.add_argument("--salida", required=True, metavar="ARCHIVO",
                   help="destino (.png, .pdf, .svg, .eps)")
    p.add_argument("--preajuste", default="predeterminado",
                   help="acs, acs-doble, rsc, elsevier, nature, aps, wiley, "
                        "tesis, presentacion, poster, grises, cascada")
    p.add_argument("--normalizar", default=None,
                   choices=["max", "minmax", "0-100", "area"])
    p.add_argument("--desplazar", type=float, default=0.0, metavar="FRACCION",
                   help="desplazamiento vertical entre curvas, como fracción "
                        "del recorrido de la mayor")
    p.add_argument("--escala", default="linear",
                   choices=["linear", "log", "sqrt", "symlog"])
    p.add_argument("--limites", nargs=2, type=float, default=None,
                   metavar=("BAJO", "ALTO"))
    p.add_argument("--marcar", nargs="*", type=float, default=None,
                   metavar="X", help="líneas verticales de referencia")
    p.add_argument("--laser", type=float, default=None, metavar="NM")
    p.add_argument("--datos", default=None, metavar="ARCHIVO",
                   help="guardar también los números dibujados")
    p.set_defaults(func=cmd_figura)

    p = sub.add_parser(
        "exportar", help="convertir una medida a otro formato"
    )
    p.add_argument("archivo")
    p.add_argument("salida", help=".csv, .tsv, .json, .md, .tex, .html, "
                                 ".jdx (JCAMP-DX) o .xy")
    p.add_argument("--laser", type=float, default=None, metavar="NM")
    p.set_defaults(func=cmd_exportar)

    p = sub.add_parser(
        "proyecto", help="crear o inspeccionar un archivo de proyecto (.rcproj)"
    )
    p.add_argument("accion", choices=["crear", "ver"])
    p.add_argument("origen", help="carpeta de medidas (crear) o .rcproj (ver)")
    p.add_argument("destino", nargs="?", default=None,
                   help="archivo .rcproj a escribir (solo con «crear»)")
    p.add_argument("--laser", type=float, default=None, metavar="NM")
    p.add_argument("--notas", default=None)
    p.set_defaults(func=cmd_proyecto)

    p = sub.add_parser(
        "micro", help="tamaño de cristalito y microdeformación de un difractograma"
    )
    p.add_argument("patron")
    p.add_argument("--anodo", default="Cu")
    p.add_argument("--instrumento", type=float, default=0.0, metavar="GRADOS",
                   help="anchura instrumental. Sin ella, el «tamaño» que salga "
                        "incluye la resolución del equipo")
    p.add_argument("--carbono", action="store_true",
                   help="además, d₀₀₂, L_c, L_a y grado de grafitización")
    p.set_defaults(func=cmd_micro)

    p = sub.add_parser("tiempos", help="cronometrar la suite")
    p.add_argument("--rapido", action="store_true",
                   help="saltarse los refinamientos, que son los lentos")
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--json", default=None, metavar="ARCHIVO")
    p.set_defaults(func=cmd_tiempos)

    p = sub.add_parser(
        "demo-datos", help="generar difractogramas o medidas electroquímicas de prueba"
    )
    p.add_argument("tipo", choices=["drx", "echem"])
    p.add_argument("carpeta", help="carpeta donde escribirlos")
    p.set_defaults(func=cmd_demo_datos)

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
