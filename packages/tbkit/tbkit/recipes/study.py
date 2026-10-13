"""A study of any set of structures: relax, Γ modes, resonant Raman, IR and a DFT check.

What ``doped_raman`` did for the 5-7 coil (pristine, graphitic N, amine), for any systems —
0D to 3D, doped or with functional groups — described in one JSON file::

    {
      "nombre": "coils_dopadas",
      "modelo": "xu_chn",                       # a parameter set or file (or chn, sp3…)
      "kpts": "auto",                           # or [1, 1, 4]; auto: ≥ 60 Å per periodic axis
      "kT": 0.05,
      "sistemas": {
        "base":  {"estructura": "coil.extxyz"},
        "N":     {"estructura": "coil_N.extxyz",  "sobre": "base", "centros": [0],
                  "born_completas": true},
        "amina": {"estructura": "coil_NH2.extxyz", "sobre": "base", "centros": [48, 59]}
      },
      "relajar":  {"fmax": 0.02, "pasos": 500},
      "hessiana": {"delta": 0.01, "radio": 6.0},
      "raman":    {"laseres_ev": [1.96, 2.33, 2.54], "eta": 0.1, "min_cm1": 900, "delta": 0.005},
      "ir":       {"min_cm1": 50, "delta": 0.005}
    }

A system ``sobre`` another (its host) is a local change of it: the host's atoms come first,
in the host's order (a substituted atom keeps its index; added atoms go at the end), and
``centros`` are the changed atoms. Only the Hessian rows (and Born charges) of atoms within
``radio`` of the change are computed; the host's are kept elsewhere
(``modes.embedded_hessian``), and ``check`` measures that against the full model. A system
without ``sobre`` is computed in full. ``born_completas``: Born charges in full even so (a
dopant whose extra electron spreads over a gapless host: the coil's N needed it, 26 % off
embedded).

Run (resumable; files in ``out/estudios/NOMBRE/<sistema>/``)::

    python -m tbkit.recipes.study ESTUDIO.json relax   [SISTEMA]
    python -m tbkit.recipes.study ESTUDIO.json hessian SISTEMA [--part r/n]
    python -m tbkit.recipes.study ESTUDIO.json check   SISTEMA
    python -m tbkit.recipes.study ESTUDIO.json raman   SISTEMA [--part r/n]
    python -m tbkit.recipes.study ESTUDIO.json born    SISTEMA [--part r/n]
    python -m tbkit.recipes.study ESTUDIO.json dft     SISTEMA [--codigo gpaw]   # DFT request
    python -m tbkit.recipes.study ESTUDIO.json report
    python -m tbkit.recipes.study ESTUDIO.json todo                   # everything, in order

Without SISTEMA, a step runs for every system (hosts first). ``report`` writes ``report.json``
and ``spectra.npz`` (open with tbkit abrir / «Abrir resultado…»). ``dft`` writes a
``nanocarbon-dft/1`` request (docs/INTERCAMBIO_DFT.md) for the modes that matter: the
strongest Raman lines in the G and D windows, the strongest IR lines and the modes with most
weight on the change; carbonforge runs it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

DEFAULTS = {"modelo": "xu_chn", "kpts": "auto", "kT": 0.05,
            "relajar": {"fmax": 0.02, "pasos": 500},
            "hessiana": {"delta": 0.01, "radio": 6.0},
            "raman": {"laseres_ev": [1.96, 2.33, 2.54], "eta": 0.1, "min_cm1": 900.0,
                      "delta": 0.005},
            "ir": {"min_cm1": 50.0, "delta": 0.005}}
#: k points: at least this many Å per periodic axis ("auto"). The coil's phonons converged
#: with kz 4 in Lz 15 Å (60 Å; from 4 to 16 k they moved ≤ 0.01 cm⁻¹).
K_LENGTH = 60.0


class Study:
    """One study file: its settings, systems and work folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.config = {**DEFAULTS, **data}
        for key in ("relajar", "hessiana", "raman", "ir"):
            self.config[key] = {**DEFAULTS[key], **data.get(key, {})}
        self.name = data.get("nombre", self.path.stem)
        self.systems = data.get("sistemas", {})
        if not self.systems:
            raise ValueError(f"{self.path}: el estudio no tiene «sistemas».")
        for key, spec in self.systems.items():
            host = spec.get("sobre")
            if host is not None and host not in self.systems:
                raise ValueError(f"{key}: «sobre» {host!r} no es un sistema del estudio.")
            if host is not None and self.systems[host].get("sobre") is not None:
                raise ValueError(f"{key}: su base {host!r} es a su vez un cambio; el embebido "
                                 "parte de un sistema calculado completo.")
            if host is not None and not spec.get("centros"):
                raise ValueError(f"{key}: un cambio sobre {host!r} necesita «centros».")
        self.work = Path(data.get("carpeta", Path("out") / "estudios" / self.name))
        self._model = None

    # ------------------------------------------------------------------ basics
    def order(self) -> list[str]:
        """Hosts before the systems built on them."""
        return [k for k in self.systems if not self.systems[k].get("sobre")] + \
            [k for k in self.systems if self.systems[k].get("sobre")]

    def folder(self, key: str) -> Path:
        path = self.work / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model(self):
        if self._model is None:
            from ..params import load_parameters

            name = self.config["modelo"]
            try:
                self._model = load_parameters(str(self.path.parent / name)
                                              if str(name).endswith(".json") else name)
            except (FileNotFoundError, KeyError, ValueError):
                from ..gui.actions import load_model

                self._model = load_model(name)
        return self._model

    def kpts(self, atoms) -> tuple[int, int, int]:
        value = self.config["kpts"]
        if value != "auto":
            return tuple(int(k) for k in value)
        lengths = atoms.cell.lengths()
        return tuple(max(1, int(np.ceil(K_LENGTH / lengths[k])))
                     if atoms.pbc[k] and lengths[k] > 0 else 1 for k in range(3))

    def calc_factory(self, atoms):
        from ..calculator import TBCalculator

        model, kpts, kT = self.model(), self.kpts(atoms), float(self.config["kT"])
        return lambda: TBCalculator(model, kpts=kpts, kT=kT)

    # ------------------------------------------------------------------ steps
    def relaxed(self, key: str):
        path = self.folder(key) / "relaxed.extxyz"
        if path.exists():
            return read(path)
        from ase.optimize import BFGS

        from ..progress import Progress, watch_optimizer
        from ..structure_io import read_structure

        source = self.path.parent / self.systems[key]["estructura"]
        atoms = read_structure(source)
        trajectory = self.folder(key) / "relax.traj"
        if trajectory.exists() and trajectory.stat().st_size > 0:
            # Resume where a stopped run was: a 480-atom relaxation is hours, and starting
            # over from the file threw them away (the BFGS Hessian is rebuilt, the steps
            # already taken are kept).
            try:
                last = read(trajectory, index=-1)
                if len(last) == len(atoms):
                    atoms.set_positions(last.get_positions())
                    print(f"{key}: reanudando desde el último paso de {trajectory.name}")
            except Exception as error:  # noqa: BLE001  (a truncated file: start over)
                print(f"{key}: no se pudo leer {trajectory.name} ({error}); desde el inicio")
        atoms.calc = self.calc_factory(atoms)()
        settings = self.config["relajar"]
        optimizer = BFGS(atoms, logfile=str(self.folder(key) / "relax.log"),
                         trajectory=str(trajectory), append_trajectory=True)
        with Progress(None, f"{self.name}: relajando {key}",
                      status=self.folder(key) / "progreso_relax.json") as bar:
            watch_optimizer(optimizer, bar)
            converged = bool(optimizer.run(fmax=float(settings["fmax"]),
                                           steps=int(settings["pasos"])))
        atoms.calc = None
        write(path, atoms)
        (self.folder(key) / "relax.json").write_text(json.dumps(
            {"converged": converged, "steps": optimizer.get_number_of_steps(),
             "kpts": list(self.kpts(atoms)), "source": str(source)}))
        return atoms

    def host_size(self, key: str) -> int | None:
        host = self.systems[key].get("sobre")
        return None if host is None else len(self.relaxed(host))

    def check_mapping(self, key: str) -> None:
        """A change must keep its host's atoms first, in order (see the module note)."""
        host = self.systems[key].get("sobre")
        if host is None:
            return
        base, atoms = self.relaxed(host), self.relaxed(key)
        n = len(base)
        if len(atoms) < n:
            raise ValueError(f"{key} tiene menos átomos que su base {host}: un cambio por "
                             "sustitución o adición deja los de la base primero.")
        centres = set(self.systems[key]["centros"])
        bad = [i for i in range(n) if i not in centres
               and base[i].symbol != atoms[i].symbol]
        # A relaxed change moves its neighbours by tenths of Å; an index pointing at another
        # atom is off by a bond length or more (mic: an atom may wrap across the cell).
        shift = atoms[:n].copy()
        shift.positions -= base.positions
        far = np.array([np.linalg.norm(shift.cell.cartesian_positions(
            (shift.cell.scaled_positions(v[None])[0] + 0.5) % 1.0 - 0.5))
            if atoms.pbc.any() else np.linalg.norm(v) for v in shift.positions])
        moved = [i for i in range(n) if i not in centres and far[i] > 0.5]
        if bad or moved:
            raise ValueError(f"{key}: el orden de los átomos no corresponde al de {host} "
                             f"(elementos distintos fuera de «centros»: {bad[:10]}; átomos "
                             f"a más de 0.5 Å de su par en la base: {moved[:10]}). Los átomos "
                             "de la base van primero y en su orden.")

    def region(self, key: str, atoms) -> list[int]:
        n = self.host_size(key)
        if n is None:
            return list(range(len(atoms)))
        centres = list(self.systems[key]["centros"]) + list(range(n, len(atoms)))
        d = atoms.get_all_distances(mic=True)[centres].min(axis=0)
        return sorted(int(i) for i in np.flatnonzero(d <= float(self.config["hessiana"]["radio"])))

    def hessian(self, key: str, part=(0, 1)):
        from ..modes import embedded_hessian, hessian_rows

        self.check_mapping(key)
        atoms = self.relaxed(key)
        rows = hessian_rows(atoms, self.calc_factory(atoms), self.folder(key) / "fd",
                            self.region(key, atoms), float(self.config["hessiana"]["delta"]),
                            part)
        if rows is None:
            return None
        n = self.host_size(key)
        if n is None:
            return 0.5 * (rows + rows.T)
        reference = self.hessian(self.systems[key]["sobre"])
        if reference is None:
            raise RuntimeError(f"Falta la hessiana de {self.systems[key]['sobre']}.")
        return embedded_hessian(reference, n, len(atoms), self.region(key, atoms), rows)

    def vibrations(self, key: str):
        from ..modes import Vibrations

        path = self.folder(key) / "hessian.npy"
        atoms = self.relaxed(key)
        if not path.exists():
            h = self.hessian(key)
            if h is None:
                raise RuntimeError(f"{key}: hessiana incompleta (corre el paso hessian).")
            np.save(path, h)
        return Vibrations.from_hessian(atoms, np.load(path),
                                       source=f"{self.config['modelo']} {self.name}/{key}")

    def core(self, key: str, atoms) -> list[int]:
        n = self.host_size(key)
        if n is None:
            return []
        return sorted(set(self.systems[key]["centros"]) | set(range(n, len(atoms))))

    def check(self, key: str) -> list[dict]:
        """Embedded modes against the full model (Rayleigh quotient, 4 modes with most weight
        on the change). Nothing to check for a system computed in full."""
        from ..modes import participation
        from ..sites import projected_frequency

        if self.host_size(key) is None:
            return []
        vib = self.vibrations(key)
        share = participation(vib, {"core": self.core(key, vib.atoms)})["core"]
        picks = [int(k) for k in np.argsort(-share)[:4]]
        out = [{"mode": k, "embedded_cm1": float(vib.frequencies[k]),
                "full_cm1": projected_frequency(vib.atoms, vib.modes[k],
                                                self.calc_factory(vib.atoms),
                                                max_disp=0.01)["frequency_cm1"],
                "share_on_core": float(share[k])} for k in picks]
        (self.folder(key) / "check.json").write_text(json.dumps(out, indent=1))
        return out

    def raman(self, key: str, part=(0, 1)):
        from ..progress import Progress
        from ..resonance import resonant_raman

        vib = self.vibrations(key)
        settings = self.config["raman"]
        chosen = [int(k) for k in np.flatnonzero(vib.frequencies > float(settings["min_cm1"]))]
        chosen = [k for k in chosen if k % part[1] == part[0]]
        with Progress(len(chosen), f"{self.name}: Raman {key}, parte {part[0] + 1}/{part[1]}",
                      status=self.folder(key) / f"progreso_raman_{part[0]}.json") as bar:
            return resonant_raman(vib.atoms, self.model(), tuple(settings["laseres_ev"]),
                                  eta=float(settings["eta"]),
                                  kmesh=max(self.kpts(vib.atoms)),   # per periodic axis
                                  kT=float(self.config["kT"]),
                                  phonons=(vib.frequencies, vib.modes), select=chosen,
                                  cache_dir=self.folder(key) / "tensors",
                                  delta=float(settings["delta"]), progress=bar.callback())

    def born_region(self, key: str, atoms) -> list[int]:
        if self.systems[key].get("born_completas") or self.host_size(key) is None:
            return list(range(len(atoms)))
        return self.region(key, atoms)

    def born(self, key: str, part=(0, 1)):
        from ..infrared import born_rows

        atoms = self.relaxed(key)
        reg = self.born_region(key, atoms)
        rows = born_rows(atoms, self.calc_factory(atoms), self.folder(key) / "born", reg,
                         float(self.config["ir"]["delta"]), part)
        if rows is None or len(reg) == len(atoms):
            return rows
        reference = self.born(self.systems[key]["sobre"])
        if reference is None:
            raise RuntimeError(f"Faltan las cargas de Born de {self.systems[key]['sobre']}.")
        out = np.zeros((len(atoms), 3, 3))
        out[:self.host_size(key)] = reference
        out[reg] = rows
        return out

    def born_ready(self, key: str) -> bool:
        keys = [key] + ([self.systems[key]["sobre"]] if self.systems[key].get("sobre") else [])
        for k in keys:
            atoms = self.relaxed(k)
            if not all((self.folder(k) / "born" / f"z_{a:04d}_{c}_{t}.npy").exists()
                       for a in self.born_region(k, atoms) for c in range(3)
                       for t in ("p", "m")):
                return False
        return True

    def infrared(self, key: str) -> dict:
        from ..infrared import mode_intensities

        vib = self.vibrations(key)
        z = self.born(key)
        if z is None:
            raise RuntimeError(f"{key}: cargas de Born incompletas.")
        keep = np.flatnonzero(vib.frequencies > float(self.config["ir"]["min_cm1"]))
        intensities, info = mode_intensities(z, vib.modes[keep])
        return {"frequencies": vib.frequencies[keep], "intensities": intensities,
                "mode_indices": keep, **info}

    # ------------------------------------------------------------------ report and DFT
    def report(self) -> dict:
        from ..modes import participation
        from ..raman import RamanResult, spectrum
        from ..sites import ring_breathing

        lasers = list(self.config["raman"]["laseres_ev"])
        out = {"titulo": f"Estudio {self.name}: Raman, banda D e IR",
               "receta": "tbkit.recipes.study", "estudio": str(self.path),
               "sistemas": [k for k in self.order()], "lasers_eV": lasers,
               "min_cm1": self.config["raman"]["min_cm1"],
               "radius_A": self.config["hessiana"]["radio"], "modelo": self.config["modelo"]}
        grid, ir_grid = np.linspace(900, 3700, 2801), np.linspace(0, 3700, 3701)
        spectra = {}
        for key in self.order():
            vib = self.vibrations(key)
            result = self.raman(key)
            atoms = vib.atoms
            symbols = atoms.get_chemical_symbols()
            groups = {"cambio": self.core(key, atoms)} if self.core(key, atoms) else {}
            for element in sorted(set(symbols) - {"C"}):
                groups[element] = [i for i, s in enumerate(symbols) if s == element]
            shares = participation(vib, groups) if groups else {}
            entry = {"atoms": len(atoms), "modes_computed": len(result.frequencies),
                     "kpts": list(self.kpts(atoms))}
            breathing = ring_breathing(atoms, vib.modes[result.mode_indices])["B"]
            entry["breathing_B"] = breathing.tolist()
            entry["raman_frequencies_cm1"] = result.frequencies.tolist()
            for li, laser in enumerate(lasers):
                r = RamanResult(result.frequencies, result.activities[li],
                                result.depolarization[li], None, None, "", result.frequencies)
                _, curve = spectrum(r, grid, fwhm=10.0, laser_nm=1239.84193 / laser)
                spectra[f"{key}_{laser:.2f}"] = curve
                rb = RamanResult(result.frequencies, result.activities[li] * breathing,
                                 result.depolarization[li], None, None, "", result.frequencies)
                _, d_curve = spectrum(rb, grid, fwhm=10.0, laser_nm=1239.84193 / laser)
                spectra[f"{key}_{laser:.2f}_breathing"] = d_curve
                entry[f"activities_{laser:.2f}"] = result.activities[li].tolist()
                order = np.argsort(-result.activities[li])[:12]
                entry[f"strongest_{laser:.2f}"] = [
                    {"frequency_cm1": float(result.frequencies[j]),
                     "activity": float(result.activities[li, j]),
                     "mode": int(result.mode_indices[j]),
                     **{f"on_{g}": float(v[result.mode_indices[j]]) for g, v in shares.items()}}
                    for j in order]
            if self.born_ready(key):
                ir = self.infrared(key)
                half = 5.0
                shape = half / np.pi / ((ir_grid[:, None] - ir["frequencies"][None]) ** 2
                                        + half ** 2)
                spectra[f"{key}_ir"] = shape @ ir["intensities"]
                order = np.argsort(-ir["intensities"])[:12]
                entry["ir"] = {"axes": ir["axes"],
                               "sum_rule_residual_e": ir["sum_rule_residual_e"],
                               "frequencies_cm1": ir["frequencies"].tolist(),
                               "intensities_km_mol": ir["intensities"].tolist(),
                               "strongest": [{"frequency_cm1": float(ir["frequencies"][j]),
                                              "intensity_km_mol": float(ir["intensities"][j]),
                                              "mode": int(ir["mode_indices"][j])}
                                             for j in order]}
            out[key] = entry
        np.savez(self.work / "spectra.npz", grid=grid, ir_grid=ir_grid, **spectra)
        (self.work / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return out

    def dft_modes(self, key: str, report: dict | None = None) -> list[int]:
        """The modes worth a DFT check: 3 strongest Raman lines in G (1500–1700 cm⁻¹) and 2 in
        D (1100–1450, weighted by ring breathing) at the middle laser, 3 strongest IR lines,
        and the 3 modes with most weight on the change."""
        from ..modes import participation

        report = report or json.loads((self.work / "report.json").read_text())
        entry = report[key]
        lasers = report["lasers_eV"]
        tag = f"{lasers[len(lasers) // 2]:.2f}"
        w = np.array(entry["raman_frequencies_cm1"])
        a = np.array(entry[f"activities_{tag}"])
        b = np.array(entry["breathing_B"])
        vib = self.vibrations(key)
        indices = np.flatnonzero(vib.frequencies > float(self.config["raman"]["min_cm1"]))
        picks = []
        for mask, weight, n in (((w > 1500) & (w < 1700), a, 3),
                                ((w > 1100) & (w < 1450), a * b, 2)):
            order = [j for j in np.argsort(-weight) if mask[j]][:n]
            picks += [int(indices[j]) for j in order]
        if "ir" in entry:
            picks += [s["mode"] for s in entry["ir"]["strongest"][:3]]
        core = self.core(key, vib.atoms)
        if core:
            share = participation(vib, {"core": core})["core"]
            share[vib.frequencies < 100] = 0
            picks += [int(k) for k in np.argsort(-share)[:3]]
        return sorted(set(picks))

    def dft_request(self, key: str, code: str = "gpaw", relax: bool = True,
                    mpi: int = 4) -> Path:
        """A DFT request (nanocarbon-dft/1) for the system's important modes."""
        from .. import dft_exchange as dx

        modes = self.dft_modes(key)
        folder = self.work / "dft" / key
        return dx.mode_check_request(folder, self.vibrations(key), modes, code,
                                     relax_first=relax, resources={"mpi": mpi, "hilos": 1})


STEPS = ("relax", "hessian", "check", "raman", "born", "dft", "report", "todo")
#: Steps that split in parts (--part r/N); the others run as one process.
PARALLEL_STEPS = ("hessian", "raman", "born")


def run_step(study: Study, step: str, keys: list[str], part=(0, 1), **kwargs) -> None:
    if step == "report":
        print(json.dumps({k: v for k, v in study.report().items()
                          if not isinstance(v, dict)}, indent=1, ensure_ascii=False))
        return
    if step == "todo":
        for s in ("relax", "hessian", "check", "raman", "born"):
            run_step(study, s, keys, part)
        run_step(study, "report", keys)
        return
    for key in keys:
        if step == "relax":
            study.relaxed(key)
            print(f"{key}: relajada")
        elif step == "hessian":
            done = study.hessian(key, part)
            if done is not None:
                np.save(study.folder(key) / "hessian.npy", done)
            print(f"{key}: hessiana {'completa' if done is not None else f'parte {part} hecha'}")
        elif step == "check":
            for row in study.check(key):
                print(f"{key}: modo {row['mode']} embebido {row['embedded_cm1']:.1f} · completo "
                      f"{row['full_cm1']:.1f} cm⁻¹")
        elif step == "raman":
            study.raman(key, part)
            print(f"{key}: Raman parte {part} hecha")
        elif step == "born":
            done = study.born(key, part)
            print(f"{key}: Born {'completas' if done is not None else f'parte {part} hecha'}")
        elif step == "dft":
            print(f"{key}: solicitud DFT en {study.dft_request(key, **kwargs)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("estudio", type=Path, help="archivo JSON del estudio")
    parser.add_argument("step", choices=STEPS,
                        help="paso: relax, hessian, check, raman, born, dft, report o todo")
    parser.add_argument("sistema", nargs="?", default=None,
                        help="un sistema del estudio (vacío: todos, las bases primero)")
    parser.add_argument("--part", default="0/1", help="r/n: solo la parte r de n")
    parser.add_argument("--codigo", default="gpaw", choices=("gpaw", "qe", "dftb+"),
                        help="código DFT del paso dft")
    parser.add_argument("--sin-relajar-dft", action="store_true",
                        help="el paso dft no relaja con DFT antes de los modos")
    parser.add_argument("--mpi", type=int, default=4, help="procesos MPI pedidos al DFT")
    args = parser.parse_args(argv)
    study = Study(args.estudio)
    keys = [args.sistema] if args.sistema else study.order()
    unknown = [k for k in keys if k not in study.systems]
    if unknown:
        raise SystemExit(f"sistemas desconocidos {unknown}; hay: {', '.join(study.systems)}")
    part = tuple(int(x) for x in args.part.split("/"))
    extra = {"code": args.codigo, "relax": not args.sin_relajar_dft, "mpi": args.mpi} \
        if args.step == "dft" else {}
    run_step(study, args.step, keys, part, **extra)


if __name__ == "__main__":
    main()
