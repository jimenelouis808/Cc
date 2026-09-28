"""``tbkit``: tight binding from the command line.

    tbkit levels  cinta.xyz                       # niveles, gap, cargas
    tbkit levels  cinta.xyz --model sp3 --scc     # sp3 de carbono, cargas autoconsistentes
    tbkit raman   piridina.xyz --model chn        # C/H/N (Xu + H, N ajustados a GPAW)
    tbkit raman   piridina.xyz --model chn --modes dynmat.out   # fonones de QE, α de TB
    tbkit raman   butadieno.xyz --model chn --resonant 2.33 3.5 4.0  # perfiles resonantes
    tbkit bands   grafeno.xyz --path GKMG -o bandas.csv
    tbkit dos     tubo.xyz --kmesh 60 --pdos element -o dos.csv
    tbkit hubbard zgnr.xyz --U 2.7 --kmesh 48 --m-energy m.csv
    tbkit hubbard flake.xyz --field 0 0.5 11 -o campo.csv
    tbkit orbital benceno.xyz --band homo -o homo.cube
    tbkit relax   cluster.xyz --model sp3 -o relajado.extxyz
    tbkit phonons diamante.extxyz --model sp3 --kmesh 8
    tbkit raman   diamante.extxyz --model sp3 --kmesh 8 -o raman.csv
    tbkit run     simulacion.json                  # reproducible: guarda un registro
    tbkit gpaw-levels calc/gpaw.txt

Structures: any file ASE reads (extxyz from carbonforge or nanocarbon_lab
keeps the cell and periodicity). Models: ``--model pi`` (default), ``sp3``
(Xu carbon), ``chn`` (Xu carbon plus H and N fitted to GPAW, SCC),
``--parameters FILE.json`` (any parameter file, e.g. your own fit), or
``--skf DIR --orbitals "C=s,px,py,pz H=s"``. Charges are self-consistent
when the model is (chn, .skf); ``--scc``/``--no-scc`` overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _model(args):
    from .params import load_parameters, pi_model, xu_carbon
    from .skf import load_skf_set

    if getattr(args, "parameters", None):
        return load_parameters(args.parameters)
    if args.skf:
        orbitals = {}
        for item in args.orbitals.split():
            element, names = item.split("=")
            orbitals[element] = tuple(names.split(","))
        return load_skf_set(args.skf, orbitals)
    if args.model == "sp3":
        return xu_carbon()
    if args.model == "chn":
        return load_parameters("xu_chn")
    return pi_model(t=args.t)


def _system(args):
    from ase.io import read

    from .hamiltonian import System

    atoms = read(args.structure)
    return System.build(atoms, _model(args))


def _kpoints(system, n):
    from .kpoints import gamma, mesh

    return mesh(system.atoms, n) if system.periodic else gamma()


def _write_table(path: Optional[str], header: list[str], columns: list[np.ndarray]) -> None:
    rows = np.column_stack(columns)
    if path is None:
        print("\t".join(header))
        for row in rows[:: max(1, len(rows) // 40)]:
            print("\t".join(f"{v:.6g}" for v in row))
        return
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")
    print(f"→ {path}")


def cmd_levels(args) -> int:
    from .analysis import atomic_charges, bond_orders
    from .scc import self_consistent
    from .solver import solve

    system = _system(args)
    if system.model.scc if args.scc is None else args.scc:
        result = self_consistent(system, charge=args.charge)
        solution, charges = result.solution, result.charges
        print(result.summary())
    else:
        solution = solve(system, *_kpoints(system, args.kmesh), charge=args.charge)
        charges = atomic_charges(solution)
    homo, lumo = solution.homo_lumo()
    print(f"Modelo: {system.model.name}; {system.basis.size} orbitales, "
          f"{solution.electrons:g} electrones")
    print(f"HOMO {homo:.4f} eV, LUMO {lumo:.4f} eV, gap {solution.gap():.4f} eV, "
          f"E_F {solution.fermi:.4f} eV")
    if not system.periodic:
        levels = np.sort(solution.energies[0, 0])
        print("Niveles (eV):", " ".join(f"{e:.3f}" for e in levels))
    top = sorted(charges.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("Cargas (e, + = pierde electrones):",
          ", ".join(f"{system.atoms[a].symbol}{a} {q:+.3f}" for a, q in top))
    if not system.periodic and args.bonds:
        for (a, b), value in sorted(bond_orders(solution).items(), key=lambda kv: -kv[1])[:12]:
            print(f"  enlace {a}-{b}: {value:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": system.model.name, "homo": homo, "lumo": lumo, "gap": solution.gap(),
            "fermi": solution.fermi, "charges": {str(a): q for a, q in charges.items()},
            "levels": np.sort(solution.energies.ravel()).tolist()}, indent=1))
        print(f"→ {args.json}")
    return 0


def cmd_bands(args) -> int:
    from .analysis import bands
    from .kpoints import band_path

    system = _system(args)
    if not system.periodic:
        print("Las bandas necesitan una estructura periódica (pbc en el archivo).")
        return 1
    path = band_path(system.atoms, args.path, args.npoints)
    energies = bands(system, path)[0]
    x, _, _ = path.get_linear_kpoint_axis()
    _write_table(args.out, ["k"] + [f"banda{i}" for i in range(energies.shape[1])],
                 [x] + [energies[:, i] for i in range(energies.shape[1])])
    return 0


def cmd_dos(args) -> int:
    from .analysis import dos, pdos
    from .solver import solve

    system = _system(args)
    solution = solve(system, *_kpoints(system, args.kmesh))
    grid = np.linspace(solution.energies.min() - 1, solution.energies.max() + 1, args.points)
    grid, total = dos(solution, grid, args.sigma)
    columns, header = [grid - solution.fermi, total], ["E-E_F", "DOS"]
    if args.pdos:
        _, parts = pdos(solution, args.pdos, grid, args.sigma)
        for label, values in parts.items():
            header.append(label)
            columns.append(values)
    print(f"E_F = {solution.fermi:.4f} eV")
    _write_table(args.out, header, columns)
    return 0


def cmd_hubbard(args) -> int:
    from .hubbard import (
        magnetization_vs_doping,
        magnetization_vs_energy,
        magnetization_vs_field,
        mean_field,
    )

    system = _system(args)
    k, w = _kpoints(system, args.kmesh)
    options = dict(U=args.U, kpts=k, weights=w, kT=args.kT)
    if args.field:
        start, stop, n = args.field
        h, m, chi, _ = magnetization_vs_field(system, np.linspace(start, stop, int(n)),
                                              guess=args.guess, **options)
        _write_table(args.out, ["h_eV", "M_muB", "chi_muB_per_eV"], [h, m, chi])
        return 0
    if args.doping:
        start, stop, n = args.doping
        x, m, _ = magnetization_vs_doping(system, charges=np.linspace(start, stop, int(n)),
                                          guess=args.guess, **options)
        _write_table(args.out, ["carga_e", "M_muB"], [x, m])
        return 0
    result = mean_field(system, guess=args.guess, **options)
    print(result.summary())
    moments = result.moments
    top = sorted(moments.items(), key=lambda kv: -abs(kv[1]))[:10]
    print("Momentos locales (μB):",
          ", ".join(f"{system.atoms[a].symbol}{a} {m:+.3f}" for a, m in top))
    if args.m_energy:
        e, m, dm = magnetization_vs_energy(result, sigma=args.sigma)
        _write_table(args.m_energy, ["E-E_F", "m_muB", "dm_dE"], [e, m, dm])
    return 0 if result.converged else 2


def cmd_orbital(args) -> int:
    from .analysis import orbital_on_grid, write_cube
    from .solver import solve

    system = _system(args)
    if system.periodic:
        print("Orbitales en rejilla: solo sistemas finitos.")
        return 1
    solution = solve(system)
    occupied = int(round(solution.electrons / 2))
    band = {"homo": occupied - 1, "lumo": occupied}.get(args.band.lower()) \
        if not args.band.lstrip("-").isdigit() else int(args.band)
    origin, steps, values = orbital_on_grid(solution, band, spacing=args.spacing)
    energy = solution.energies[0, 0, band]
    write_cube(args.out, system.atoms, origin, steps, values,
               comment=f"tbkit banda {band} ({energy:.4f} eV)")
    print(f"Banda {band}: {energy:.4f} eV → {args.out}")
    return 0


def cmd_run(args) -> int:
    from .record import run_simulation

    record = run_simulation(args.config)
    results = record["results"]
    summary = {k: v for k, v in results.items() if not isinstance(v, (list, dict))}
    print(f"Tarea {record['task']} con '{record['model']['name']}' "
          f"(tbkit {record['tbkit']}, commit {record['commit'] or 'desconocido'})")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"→ {record['_path']}")
    return 0


def cmd_relax(args) -> int:
    from ase.io import read, write

    from .tasks import relax

    atoms = read(args.structure)
    results, final = relax(atoms, _model(args), kmesh=args.kmesh, kT=args.kT, fmax=args.fmax,
                           steps=args.steps, scc=args.scc)
    print(f"{'Convergido' if results['converged'] else 'SIN CONVERGER'} en "
          f"{results['steps']} pasos: E = {results['energy']:.6f} eV, fuerza máxima "
          f"{results['max_force']:.4f} eV/Å")
    write(args.out, final)
    print(f"→ {args.out}")
    return 0 if results["converged"] else 2


def cmd_phonons(args) -> int:
    from ase.io import read

    from .tasks import phonons

    results, _ = phonons(read(args.structure), _model(args), kmesh=args.kmesh, kT=args.kT,
                         delta=args.delta, scc=args.scc)
    if results["residual_force"] > 0.05:
        print(f"AVISO: fuerza residual {results['residual_force']:.3f} eV/Å; relaja antes "
              "(tbkit relax) o las frecuencias no son las armónicas.")
    frequencies = results["frequencies_cm1"]
    print("Frecuencias Γ (cm⁻¹, negativas = imaginarias):")
    print("  " + " ".join(f"{f:.1f}" for f in frequencies))
    if args.out:
        np.savetxt(args.out, frequencies, header="frecuencia_cm-1", comments="")
        print(f"→ {args.out}")
    return 0


def cmd_raman(args) -> int:
    from ase.io import read

    from .raman import raman, raman_from_qe, spectrum

    if args.resonant:
        from .qe import modes_for_raman, read_qe_modes
        from .resonance import resonant_raman

        atoms = read(args.structure)
        model = _model(args)
        phonons = None
        if args.modes:
            modes = read_qe_modes(args.modes)
            phonons = (modes.frequencies,
                       modes_for_raman(modes, atoms.get_masses(), args.modes_kind))
        resonant = resonant_raman(atoms, model, args.resonant, eta=args.eta,
                                  kmesh=args.kmesh, kT=args.kT, delta=args.delta,
                                  phonons=phonons)
        print(resonant.summary())
        result = resonant.at(args.resonant[0])
        if args.out:
            laser = None if args.bare else 1239.84193 / args.resonant[0]
            temperature = None if args.bare else args.temperature
            grid, intensity = spectrum(result, fwhm=args.fwhm, laser_nm=laser,
                                       temperature_k=temperature)
            _write_table(args.out, ["desplazamiento_cm-1", "intensidad"], [grid, intensity])
        return 0
    options = {"kmesh": args.kmesh, "kT": args.kT, "delta": args.delta,
               "screening": args.screening}
    if args.modes:
        result = raman_from_qe(read(args.structure), _model(args), args.modes,
                               kind=args.modes_kind, **options)
    else:
        result = raman(read(args.structure), _model(args), **options)
    print(result.summary())
    if args.out:
        laser = None if args.bare else args.laser
        temperature = None if args.bare else args.temperature
        grid, intensity = spectrum(result, fwhm=args.fwhm, laser_nm=laser,
                                   temperature_k=temperature)
        _write_table(args.out, ["desplazamiento_cm-1", "intensidad"], [grid, intensity])
    return 0


def cmd_ir(args) -> int:
    from ase.io import read

    from .infrared import infrared, ir_spectrum

    atoms = read(args.structure)
    phonons = None
    if args.modes:
        from .qe import modes_for_raman, read_qe_modes

        modes = read_qe_modes(args.modes)
        phonons = (modes.frequencies, modes_for_raman(modes, atoms.get_masses(),
                                                      args.modes_kind))
    result = infrared(atoms, _model(args), phonons=phonons,
                      onsite_dipoles=not args.charges_only)
    print(result.summary())
    if args.out:
        grid, intensity = ir_spectrum(result, fwhm=args.fwhm)
        _write_table(args.out, ["numero_de_onda_cm-1", "absorcion_km_mol_cm"], [grid, intensity])
    return 0


def cmd_graphene_raman(args) -> int:
    from .graphene import graphene_raman, load_phonons

    results = graphene_raman(args.laser, load_phonons(args.phonons), gamma=args.gamma,
                             dk=args.dk, dq=args.dq, workers=args.workers)
    print(f"Grafeno prístino, fonones: {args.phonons}; γ = {args.gamma} eV")
    print(f"{'láser eV':>9} {'G cm⁻¹':>8} {'2D cm⁻¹':>8} {'2D´ cm⁻¹':>9} {'I(2D)/I(G)':>11}")
    for r in results:
        prime = r["2D'_position"]
        print(f"{r['laser_ev']:9.2f} {r['g_frequency']:8.1f} {r['2D_position']:8.0f} "
              f"{prime:9.0f} "
              f"{r['2D_intensity'] / r['g_intensity']:11.2f}")
    if len(results) > 1:
        slope = np.polyfit([r["laser_ev"] for r in results],
                           [r["2D_position"] for r in results], 1)[0]
        print(f"dispersión de la 2D: {slope:.0f} cm⁻¹/eV")
    if args.out:
        columns = [results[0]["grid"]] + [r["spectrum"] for r in results]
        _write_table(args.out, ["desplazamiento_cm-1"] + [f"I_{r['laser_ev']:.2f}eV"
                                                          for r in results], columns)
    return 0


def cmd_gpaw_levels(args) -> int:
    from .fit import read_gpaw_eigenvalues

    levels = read_gpaw_eigenvalues(args.path)
    for spin in range(levels.energies.shape[0]):
        n = levels.n_occupied(spin)
        e = levels.energies[spin]
        print(f"espín {spin}: {len(e)} niveles, {n} ocupados, HOMO {e[n - 1]:.4f}, "
              f"LUMO {e[n]:.4f} eV")
    if levels.fermi is not None:
        print(f"E_F {levels.fermi:.4f} eV")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbkit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def structure_command(name, help_text, func):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("structure")
        p.add_argument("--model", choices=("pi", "sp3", "chn"), default="pi",
                       help="pi (π Hückel), sp3 (carbono de Xu, con parte repulsiva) o chn "
                            "(Xu + H y N ajustados a GPAW, SCC).")
        p.add_argument("--parameters", default=None,
                       help="Archivo JSON de parámetros (sustituye a --model).")
        p.add_argument("--t", type=float, default=-2.7, help="Hopping π, eV (modelo pi).")
        p.add_argument("--skf", default=None, help="Carpeta con archivos A-B.skf de DFTB.")
        p.add_argument("--orbitals", default="C=s,px,py,pz H=s",
                       help="Base con --skf, p. ej. \"C=s,px,py,pz H=s\".")
        p.add_argument("--kmesh", type=int, default=24, help="Puntos k por eje periódico.")
        p.set_defaults(func=func)
        return p

    lv = structure_command("levels", "Niveles, gap y cargas.", cmd_levels)
    lv.add_argument("--charge", type=float, default=0.0)
    lv.add_argument("--scc", action=argparse.BooleanOptionalAction, default=None,
                    help="Cargas autoconsistentes (por defecto, las del modelo).")
    lv.add_argument("--bonds", action="store_true", help="Órdenes de enlace de Mayer.")
    lv.add_argument("--json", default=None)

    bd = structure_command("bands", "Bandas a lo largo de un camino.", cmd_bands)
    bd.add_argument("--path", default=None, help="Puntos especiales (p. ej. GKMG).")
    bd.add_argument("--npoints", type=int, default=200)
    bd.add_argument("-o", "--out", default=None)

    ds = structure_command("dos", "Densidad de estados (y PDOS).", cmd_dos)
    ds.add_argument("--sigma", type=float, default=0.05)
    ds.add_argument("--points", type=int, default=3000)
    ds.add_argument("--pdos", choices=("element", "atom", "orbital"), default=None)
    ds.add_argument("-o", "--out", default=None)

    hb = structure_command("hubbard", "Hubbard de campo medio: momentos y magnetización.",
                           cmd_hubbard)
    hb.add_argument("--U", type=float, default=None, help="eV (por defecto, el del modelo).")
    hb.add_argument("--kT", type=float, default=0.005)
    hb.add_argument("--guess", choices=("antiferro", "ferro", "random", "paramagnetic"),
                    default="antiferro")
    hb.add_argument("--sigma", type=float, default=0.05)
    hb.add_argument("--m-energy", default=None, help="CSV con m(E) y dm/dE.")
    hb.add_argument("--field", type=float, nargs=3, metavar=("H0", "H1", "N"), default=None,
                    help="Barrido de energía Zeeman h = μB·B (eV).")
    hb.add_argument("--doping", type=float, nargs=3, metavar=("Q0", "Q1", "N"), default=None,
                    help="Barrido de carga añadida (e).")
    hb.add_argument("-o", "--out", default=None)

    ob = structure_command("orbital", "Un orbital molecular en un archivo cube.", cmd_orbital)
    ob.add_argument("--band", default="homo", help="homo, lumo o un índice.")
    ob.add_argument("--spacing", type=float, default=0.2)
    ob.add_argument("-o", "--out", default="orbital.cube")

    rl = structure_command("relax", "Relajar posiciones (modelos con parte repulsiva).",
                           cmd_relax)
    rl.add_argument("--kT", type=float, default=0.02)
    rl.add_argument("--fmax", type=float, default=0.01)
    rl.add_argument("--steps", type=int, default=500)
    rl.add_argument("--scc", action=argparse.BooleanOptionalAction, default=None,
                    help="Cargas autoconsistentes (por defecto, las del modelo).")
    rl.add_argument("-o", "--out", default="relajada.extxyz")

    ph = structure_command("phonons", "Fonones en Γ (modelos con parte repulsiva).",
                           cmd_phonons)
    ph.add_argument("--kT", type=float, default=0.02)
    ph.add_argument("--delta", type=float, default=0.005)
    ph.add_argument("--scc", action=argparse.BooleanOptionalAction, default=None,
                    help="Cargas autoconsistentes (por defecto, las del modelo).")
    ph.add_argument("-o", "--out", default=None)

    rm = structure_command("raman", "Raman no resonante (modelo con parte repulsiva; con gap).",
                           cmd_raman)
    rm.add_argument("--kT", type=float, default=0.01)
    rm.add_argument("--delta", type=float, default=0.01)
    rm.add_argument("--screening", choices=("auto", "scc", "none"), default="auto")
    rm.add_argument("--modes", default=None,
                    help="Modos en Γ de Quantum ESPRESSO (filout/fileig de dynmat.x, "
                         "flvec/fleig de matdyn.x): frecuencias y modos de QE, α del modelo.")
    rm.add_argument("--resonant", type=float, nargs="+", default=None, metavar="EV",
                    help="Raman resonante a estas energías de láser (eV); el espectro (-o) es "
                         "el de la primera.")
    rm.add_argument("--eta", type=float, default=0.1,
                    help="Ensanchamiento de las excitaciones en resonancia (eV).")
    rm.add_argument("--modes-kind", default="auto",
                    choices=("auto", "displacements", "eigenvectors"),
                    help="Qué contiene el archivo de modos (auto: autovectores si son "
                         "ortonormales).")
    rm.add_argument("--fwhm", type=float, default=8.0)
    rm.add_argument("--laser", type=float, default=532.0, help="nm")
    rm.add_argument("--temperature", type=float, default=300.0, help="K")
    rm.add_argument("--bare", action="store_true", help="Actividades sin láser ni Bose.")
    rm.add_argument("-o", "--out", default=None, help="CSV del espectro ensanchado.")

    rn = sub.add_parser("run", help="Ejecutar un archivo de simulación y guardar su registro.")
    rn.add_argument("config")
    rn.set_defaults(func=cmd_run)

    gp = sub.add_parser("gpaw-levels", help="Niveles de un gpaw.txt (para ajustar).")
    gp.add_argument("path")
    gp.set_defaults(func=cmd_gpaw_levels)
    ir = structure_command("ir", "Intensidades IR (finitos; modelo con parte repulsiva o "
                                 "--modes).", cmd_ir)
    ir.add_argument("--modes", default=None, help="Modos en Γ de Quantum ESPRESSO.")
    ir.add_argument("--modes-kind", default="auto",
                    choices=("auto", "displacements", "eigenvectors"))
    ir.add_argument("--charges-only", action="store_true",
                    help="Dipolo solo de las cargas (sin dipolos intraatómicos).")
    ir.add_argument("--fwhm", type=float, default=10.0)
    ir.add_argument("-o", "--out", default=None, help="CSV del espectro ensanchado.")

    gr = sub.add_parser("graphene-raman",
                        help="G, 2D y 2D' del grafeno por doble resonancia (modelo π).")
    gr.add_argument("--laser", type=float, nargs="+", default=[2.41], metavar="EV")
    gr.add_argument("--phonons", default="gpaw",
                    help="gpaw (PBE, incluidos), xu (modelo de Xu) o un JSON de constantes.")
    gr.add_argument("--gamma", type=float, default=0.1, help="Ensanchamiento electrónico, eV.")
    gr.add_argument("--dk", type=float, default=0.01, help="Paso de la malla en k, 1/Å.")
    gr.add_argument("--dq", type=float, default=0.03, help="Paso de la malla en q, 1/Å.")
    gr.add_argument("--workers", type=int, default=1)
    gr.add_argument("-o", "--out", default=None, help="CSV con los espectros de segundo orden.")
    gr.set_defaults(func=cmd_graphene_raman)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
