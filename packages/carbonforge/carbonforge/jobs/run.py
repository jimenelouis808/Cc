"""Run a calculation directory described by ``job.json``, step by step.

    python -m carbonforge.jobs.run DIR [--nprocs N] [--restart]

Each step runs only after the previous one exited with code 0; its standard
output goes to the step's output file, and ``job.json`` records the start,
the end and any failure. A directory interrupted halfway resumes at the first
step that did not finish (``--restart`` starts over).

Programs are looked up on ``PATH``. A different binary can be named with an
environment variable ``CARBONFORGE_<PROGRAM>`` (``CARBONFORGE_PW_X=/opt/qe/bin/pw.x``),
and the MPI launcher with ``CARBONFORGE_MPI`` (default ``mpiexec -n {n}``).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .manifest import JobManifest, Step

#: Other names the same program is installed under.
_ALIASES = {"lmp": ("lmp", "lmp_mpi", "lmp_serial"), "siesta": ("siesta",)}


def find_program(program: str) -> Optional[str]:
    """The executable for ``program``: environment override, then ``PATH``."""
    override = os.environ.get("CARBONFORGE_" + re.sub(r"[^A-Za-z0-9]", "_", program).upper())
    if override:
        return override
    for name in _ALIASES.get(program, (program,)):
        found = shutil.which(name)
        if found:
            return found
    return None


def mpi_prefix(nprocs: int) -> list[str]:
    """The MPI launcher for ``nprocs`` processes, or nothing in serial."""
    if nprocs <= 1:
        return []
    template = os.environ.get("CARBONFORGE_MPI")
    if template:
        return shlex.split(template.format(n=nprocs))
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        raise RuntimeError("No se encontró mpiexec ni mpirun: instala OpenMPI o usa 1 proceso.")
    return [launcher, "-n", str(nprocs)]


def step_command(step: Step, nprocs: int) -> list[str]:
    """argv of one step (standard input and output are handled by the caller)."""
    if step.program == "carbonforge":
        return [sys.executable, "-m", "carbonforge.cli.main", *step.args]
    program = find_program(step.program)
    if program is None:
        raise RuntimeError(
            f"No se encontró {step.program} en el PATH. Instálalo, o indica su ruta con "
            f"CARBONFORGE_{re.sub(r'[^A-Za-z0-9]', '_', step.program).upper()}."
        )
    parallel = step.parallel and nprocs > 1
    argv = (mpi_prefix(nprocs) if parallel else []) + [program]
    # Pools must divide the processes; otherwise QE refuses to start.
    if parallel and step.pools > 1 and nprocs % step.pools == 0:
        argv += ["-nk", str(step.pools)]
    argv += list(step.args)
    return argv if step.stdin or not step.input else argv + ["-in", step.input]


class _Cancelled(Exception):
    pass


def run(directory: Path, nprocs: int = 1, restart: bool = False) -> int:
    """Run the remaining steps; returns the exit code (0 when all finished)."""
    directory = Path(directory)
    manifest = JobManifest.load(directory)
    if restart:
        manifest.reset()
    child: dict[str, Optional[subprocess.Popen]] = {"process": None}

    def on_term(_signum, _frame):
        process = child["process"]
        if process is not None and process.poll() is None:
            process.terminate()
        raise _Cancelled

    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        manifest.set_status("running", f"{nprocs} proceso(s)")
        manifest.save(directory)
        for step in manifest.steps:
            if step.done:
                continue
            argv = step_command(step, nprocs)
            print(f"[{step.name}] {' '.join(argv)}", flush=True)
            manifest.history.append({"step": step.name, "command": argv})
            manifest.save(directory)
            stdin = (directory / step.input).open("rb") if step.stdin else None
            with (directory / step.output).open("wb") as out:
                child["process"] = subprocess.Popen(argv, cwd=directory, stdin=stdin,
                                                    stdout=out, stderr=subprocess.STDOUT)
                code = child["process"].wait()
            if stdin is not None:
                stdin.close()
            if code != 0:
                manifest.set_status("error", f"{step.name} terminó con código {code} "
                                             f"(ver {step.output})")
                manifest.save(directory)
                print(manifest.history[-1]["message"], flush=True)
                return code
            step.done = True
            manifest.save(directory)
        manifest.set_status("done")
        manifest.save(directory)
        print("terminado", flush=True)
        return 0
    except _Cancelled:
        manifest.set_status("cancelled", "detenido por el usuario")
        manifest.save(directory)
        return 143
    except RuntimeError as exc:
        manifest.set_status("error", str(exc))
        manifest.save(directory)
        print(exc, flush=True)
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m carbonforge.jobs.run",
                                     description="Corre los pasos de job.json en orden.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nprocs", type=int, default=1, help="Procesos MPI (pasos paralelos).")
    parser.add_argument("--restart", action="store_true", help="Empezar desde el primer paso.")
    args = parser.parse_args(argv)
    return run(args.directory, nprocs=args.nprocs, restart=args.restart)


if __name__ == "__main__":
    sys.exit(main())
